# Cloud training (Google Cloud)

What worked for the v19 run (September 2026). Scripts are in `cloud/`.

## Machine and cost
- Project `future-sight-training`. Its global CPU quota is 32, which fits one 32-vCPU VM.
- `t2d-standard-32` (32 physical cores, 128 GB) on-demand in us-west1-a: about $1.35/hour.
  `c2d-highcpu-32` was out of stock in every us-central1 zone.
- Spot (about $0.81/hour) was preempted after 2 minutes. Nothing restarts a preempted VM, so each
  preemption stops the run until someone restarts it.
- 20 envs collect about 540 steps/s (SB3 steps envs in lockstep, so more cores barely help); the
  10M-step v19 run took about 9 hours of training, about $12.
- A stopped VM still bills its 50 GB disk (about $5/month). Delete the VM when a run is done.

## Running
1. On the Mac, from `pokemon-rl/`: `bash cloud/pack.sh` (writes `pokemon-rl.tar` and `showdown.tar`).
2. Create the VM (on a stockout, try another zone or region):
   ```
   gcloud compute instances create fs-train --zone=us-west1-a --machine-type=t2d-standard-32 \
     --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB --boot-disk-type=pd-balanced
   ```
3. Upload and set up (a few minutes):
   ```
   gcloud compute scp pokemon-rl.tar showdown.tar fs-train:~ --zone=us-west1-a
   gcloud compute ssh fs-train --zone=us-west1-a --command "tar xf pokemon-rl.tar && bash pokemon-rl/cloud/vm_setup.sh"
   ```
4. Train inside tmux. `VERSION` must match `CONFIG["version"]`. The VM powers itself off when
   training ends, stalls for 20 minutes, or passes `STOP_AT`:
   ```
   gcloud compute ssh fs-train --zone=us-west1-a --command "tmux new -d -s train 'VERSION=v20 TIMESTEPS=20000000 STOP_AT=10000000 bash ~/pokemon-rl/cloud/run_training.sh'"
   ```
5. Check progress:
   ```
   gcloud compute ssh fs-train --zone=us-west1-a --command "grep -a total_timesteps ~/train_v20.log | tail -1; grep -a '>>> Eval @' ~/train_v20.log | tail -2"
   ```
6. Once it has powered off: `gcloud compute instances start fs-train --zone=us-west1-a`, then
   `VERSION=v20 bash cloud/fetch_results.sh`, check the files, then
   `gcloud compute instances delete fs-train --zone=us-west1-a`.

To resume after a stop, start the VM and repeat step 4 with the same `TIMESTEPS` (the learning
rate schedule depends on it); training continues from the last checkpoint (every 200K steps).
`~/<VERSION>_DONE` or `~/<VERSION>_STOPPED_AT_<steps>` on the VM says how a run ended.

## Pitfalls from the v19 run
- macOS tar adds `._*` files unless `COPYFILE_DISABLE=1`, and the Showdown build fails on them.
  `pack.sh` sets it; `vm_setup.sh` deletes any that slip through.
- On Linux the training script must use `forkserver` (poke-env's event loop doesn't survive
  `fork`) and one torch thread per env process; both are in the script now.
- Reusing one eval env across evaluations hung training at the second eval; each evaluation now
  gets its own env.
- Hourly checks from a Claude session only run while the Mac and the session are awake and idle.
  The stall guard in `run_training.sh` is what protects the budget.
- `gcloud compute scp` dropped a 2.4 GB fetch partway; `fetch_results.sh` uses rsync, which
  resumes and retries.
