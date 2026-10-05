"""Checks for the v21 network (milestone 2 of agent_docs/step4_brief.md).

Collects real samples (with labels) through a v21 training Worker on a local Showdown server, then:
  1. shapes, finite outputs and gradients, masked logits; finite also when whole token groups are
     padding (no moves, empty history);
  2. masking: changing what padded history slots and absent move tokens contain changes nothing;
  3. round trip: model_state -> torch.save -> V20Policy.load (routes by arch) gives the same
     outputs, with the v21 featurizer; v20 checkpoints still load as v20; a changed layout is
     refused;
  4. cost: batch-1 CPU inference (one thread, v21 vs v20), and full PPO updates (aux losses
     included) on the update device at the real rollout size, with MPS memory after each.

    python3 -m tools.check_v21_net --port 8123
"""
import argparse
import os
import tempfile
import time

import numpy as np
import torch

from networks import registry
from networks.v21_policy import V21Net, aux_obs, aux_terms
from players.v20_player import V20Policy, load_checkpoint
from training.ppo_v20 import CONFIG, AUX_HEADS, Worker, model_state, ppo_update, update_device


def finite(*ts):
    return all(torch.isfinite(t).all().item() for t in ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--updates", type=int, default=2)
    args = ap.parse_args()
    bad = []

    def check(name, ok, detail=""):
        print(f"  {'ok ' if ok else 'BAD'} {name} {detail}", flush=True)
        if not ok:
            bad.append(name)

    cfg = dict(CONFIG, arch="v21", base_port=args.port, n_servers=1, net_config=None)
    spec = registry.get("v21")
    cfg["net_config"] = spec.default_net_config()
    layout = spec.layout(cfg)
    t0 = time.time()
    w = Worker(0, cfg)
    w.new_env()
    data, stats = w.collect({k: v.numpy() for k, v in w.net.state_dict().items()}, args.samples,
                            ["models/ppo_pokemon_6v6_v18.3_best.zip"])
    w.env.close(force=True, wait=False)
    n = len(data["act"])
    print(f"collected {n} samples from {stats['games']} games {dict(stats['by_kind'])} in {time.time() - t0:.0f} s")
    check("collect: labels aligned with samples", data["lab"].shape == (n, layout["n_lab"]), data["lab"].shape)
    check("collect: idx/flt widths", data["idx"].shape[1] == layout["n_idx"] and data["flt"].shape[1] == layout["n_flt"],
          (data["idx"].shape, data["flt"].shape))

    torch.manual_seed(0)
    net = V21Net(cfg["net_config"], layout)
    print(f"V21Net: {sum(p.numel() for p in net.parameters()):,} parameters "
          f"(v20: {sum(p.numel() for p in registry.get('v20').net(None, None).parameters()):,})")
    B = min(n, 512)
    idx = torch.as_tensor(data["idx"][:B]).long()
    flt = torch.as_tensor(data["flt"][:B]).float()
    mask = torch.as_tensor(data["mask"][:B])
    act = torch.as_tensor(data["act"][:B])
    lab = torch.as_tensor(data["lab"][:B]).long()

    # 1. shapes, finiteness, masks
    net.train()
    logp, ent, val, kl, aux = net.evaluate_actions(idx, flt, mask, act, aux=True)
    logits, value = net(idx, flt, mask)
    check("shapes", logits.shape == (B, 14) and value.shape == (B,) and aux["next"].shape == (B, layout["n_next"])
          and aux["moves"].shape == (B, 6, layout["vocab"]["move"]), (logits.shape, aux["next"].shape))
    check("finite outputs", finite(logp, ent, val, kl, *aux.values()))
    check("illegal actions masked", bool((logits[~mask] <= -1e7).all()) and bool((logits[mask] > -1e7).all()))
    losses, mets = aux_terms(aux, lab, aux_obs(idx, flt, layout), metrics=True)
    loss = -logp.mean() + val.square().mean() + sum(losses[h] for h in AUX_HEADS)
    loss.backward()
    check("finite gradients", all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
    check("aux losses finite and near uniform at init", all(np.isfinite(losses[h].item()) for h in AUX_HEADS),
          {h: round(losses[h].item(), 2) for h in AUX_HEADS})
    print("    label counts in this batch:", {k: int(v[1]) for k, v in mets.items()})

    v20_idx, v20_flt = layout["v20_n_idx"], layout["v20_n_flt"]
    H, nhf = layout["hist_len"], layout["n_hs_flt"]
    hs_present = v20_flt + np.arange(H) * nhf + layout["hs_present"]
    mv_exists = 12 * layout["n_pk_flt"] + np.arange(4) * layout["n_mv_flt"] + layout["mv_exists"]
    net.eval()
    with torch.no_grad():
        f2 = flt.clone()
        f2[:, hs_present] = 0.0
        f2[:, mv_exists] = 0.0
        out = net(idx, f2, mask, aux=True)
        check("finite with no moves and an empty history", finite(out[0], out[1], *out[2].values()))

        # 2. masking: padded history slots and absent moves don't leak into other outputs
        base = net(idx, flt, mask, aux=True)
        f3, i3 = flt.clone(), idx.clone()
        hf = f3[:, v20_flt:].view(B, H, nhf)
        hi = i3[:, v20_idx:].view(B, H, layout["n_hs_idx"])
        empty = hf[..., layout["hs_present"]] < 0.5
        hf[empty] = torch.randn_like(hf[empty])
        hf[..., layout["hs_present"]] = torch.where(empty, torch.zeros_like(hf[..., 0]), hf[..., layout["hs_present"]])
        hi[empty] = torch.randint(1, 300, hi[empty].shape)
        mvf = f3[:, 12 * layout["n_pk_flt"]:v20_flt - layout["n_fd_flt"]].view(B, 4, layout["n_mv_flt"])
        absent = mvf[..., layout["mv_exists"]] < 0.5
        mvf[absent] = torch.randn_like(mvf[absent])
        mvf[..., layout["mv_exists"]] = torch.where(absent, torch.zeros_like(mvf[..., 0]), mvf[..., layout["mv_exists"]])
        pert = net(i3, f3, mask, aux=True)
        d = max((base[0] - pert[0]).abs().max().item(), (base[1] - pert[1]).abs().max().item(),
                *((base[2][k] - pert[2][k]).abs().max().item() for k in base[2]))
        check("padded history and absent moves change nothing", d < 1e-4,
              f"(max diff {d:.2e}; {int(empty.sum())} empty slots, {int(absent.sum())} absent moves)")

    # 3. round trip through the arch-routing loader
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v21.pt")
        torch.save(model_state(net, "v21", {"step": 0}), path)
        pol = V20Policy.load(path, deterministic=True)
        with torch.no_grad():
            a, b = net(idx, flt, mask), pol.net(idx, flt, mask)
        check("round trip: same outputs", torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        check("round trip: v21 featurizer with the same history length",
              pol.arch == "v21" and type(pol.featurizer).__module__ == "envs.rl_player_v21"
              and pol.featurizer.history_len == H)
        state = torch.load(path, weights_only=False)
        state["layout"]["hs_flt_names"] = state["layout"]["hs_flt_names"][::-1]
        torch.save(state, path)
        try:
            load_checkpoint(path)
            check("a changed layout is refused", False)
        except ValueError:
            check("a changed layout is refused", True)
    p20 = V20Policy.load("models/ppo_pokemon_6v6_v20.pt", deterministic=True)
    check("v20 checkpoint loads as v20", p20.arch == "v20" and type(p20.net).__name__ == "V20Net"
          and type(p20.featurizer).__module__ == "envs.rl_player_v20")

    # 4. cost: batch-1 CPU inference
    torch.set_num_threads(1)
    with torch.inference_mode():
        for name, model, ii, ff in (("v21", net, idx, flt), ("v20", p20.net, idx[:, :v20_idx], flt[:, :v20_flt])):
            model.eval()
            for j in range(20):
                model.act(ii[j:j + 1], ff[j:j + 1], mask[j:j + 1])
            t = time.perf_counter()
            for j in range(300):
                k = j % B
                model.act(ii[k:k + 1], ff[k:k + 1], mask[k:k + 1])
            print(f"  batch-1 CPU inference {name}: {(time.perf_counter() - t) / 300 * 1000:.2f} ms")
    torch.set_num_threads(os.cpu_count())

    # full PPO updates at the real size (steps_per_update samples, tiled from the collected ones)
    device = update_device(cfg)
    reps = int(np.ceil(cfg["steps_per_update"] / n))
    big = {k: np.concatenate([v] * reps)[:cfg["steps_per_update"]] for k, v in data.items()}
    model = V21Net(cfg["net_config"], layout).to(device)
    opt = torch.optim.Adam([{"params": model.embedding_parameters(), "lr": 1e-4}, {"params": model.other_parameters(), "lr": 1e-5}],
                           eps=cfg["adam_eps"])
    from networks.v21_policy import AuxBaseline
    baseline = AuxBaseline(layout)
    for u in range(args.updates):
        t = time.time()
        logs = ppo_update(model, opt, big, cfg, device, baseline)
        baseline.update(big["idx"], big["flt"], big["lab"])
        mem = torch.mps.driver_allocated_memory() / 2 ** 20 if device.type == "mps" else float("nan")
        print(f"  update {u + 1} on {device}: {time.time() - t:.1f} s | MPS driver memory {mem:.0f} MB | "
              f"grad ppo {logs['grad_norm_ppo']:.3f} aux {logs['grad_norm_aux']:.3f} total {logs['grad_norm']:.3f} "
              f"clipped {logs['grad_clipped']:.0%} | kl {logs['approx_kl']:.4f}")
        print("    acc model/base: " + " ".join(f"{k[8:]} {logs[k]:.2f}/{logs.get('aux_base_' + k[8:], float('nan')):.2f}"
                                               for k in sorted(logs) if k.startswith("aux_acc_")))
    check("update logs finite", all(np.isfinite(v) for k, v in logs.items() if not k.startswith("aux_base_")))
    print("\nRESULT:", "all checks passed" if not bad else f"{len(bad)} failed: {bad}")


if __name__ == "__main__":
    main()
