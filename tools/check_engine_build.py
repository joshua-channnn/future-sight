"""Milestone 1 of agent_docs/step6_brief.md: is the installed poke-engine the gen9 + Terastallization
build, and does its damage agree with the local Showdown fork's?

1. Tera: MCTS on a small state offers `<move>-tera` options while no Pokemon on the side has
   terastallized, and none once one has. (The PyPI wheel and the bindings' default build are gen4:
   they never offer tera.)
2. Damage: for a set of matchups covering common mechanics (type chart, STAB and Tera STAB, items,
   abilities, weather, terrain, screens, boosts, burn, special-cased moves), the fork's simulator
   (tools/showdown_damage.js: getDamage with the random roll fixed) against poke-engine's
   `calculate_damage` on the same Pokemon (stats copied from the simulator). The engine returns the
   highest roll and the highest crit roll; during search it uses 92.5% of the highest roll, with a
   separate branch when the rolls straddle a KO.

    python3 -m tools.check_engine_build
"""
import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import poke_engine as pe
from poke_env.data import to_id_str

ROOT = Path(__file__).resolve().parent.parent
SHOWDOWN = ROOT.parent / "pokemon-showdown"
WEATHER = {"": "none", "raindance": "rain", "sunnyday": "sun", "sandstorm": "sand",
           "snowscape": "snow", "desolateland": "harshsun", "primordialsea": "heavyrain"}
STATUS = {"": "none", "brn": "burn", "par": "paralyze", "psn": "poison", "tox": "toxic",
          "slp": "sleep", "frz": "freeze"}
BOOST_FIELDS = {"atk": "attack_boost", "def": "defense_boost", "spa": "special_attack_boost",
                "spd": "special_defense_boost", "spe": "speed_boost"}


def S(species, moves, ability="", item="", level=80, tera="Normal", **kw):
    return dict(species=species, moves=moves, ability=ability, item=item, level=level,
                teraType=tera, **kw)


# (description, case); every case's move is the attacker's first move
CASES = [
    ("4x, Life Orb", dict(att=S("Garchomp", ["Earthquake"], "Rough Skin", "Life Orb"),
                          def_=S("Heatran", ["Magma Storm"], "Flash Fire", "Leftovers"))),
    ("resisted, Choice Specs", dict(att=S("Dragapult", ["Draco Meteor"], "Infiltrator", "Choice Specs"),
                                    def_=S("Kingambit", ["Kowtow Cleave"], "Supreme Overlord", "Black Glasses"))),
    ("Tera STAB on a new type", dict(att=S("Dragonite", ["Extreme Speed"], "Multiscale", "Heavy-Duty Boots", tera="Normal"),
                                     def_=S("Great Tusk", ["Headlong Rush"], "Protosynthesis", "Leftovers"),
                                     att_state={"tera": True})),
    ("Tera STAB on an own type (2x)", dict(att=S("Garchomp", ["Earthquake"], "Rough Skin", "Leftovers", tera="Ground"),
                                           def_=S("Toxapex", ["Surf"], "Regenerator", "Black Sludge"),
                                           att_state={"tera": True})),
    ("rain-boosted Water", dict(att=S("Pelipper", ["Hydro Pump"], "Drizzle", "Damp Rock"),
                                def_=S("Garchomp", ["Earthquake"], "Rough Skin", "Leftovers"),
                                weather="raindance")),
    ("burned physical attacker", dict(att=S("Kingambit", ["Kowtow Cleave"], "Supreme Overlord", "Leftovers"),
                                      def_=S("Hatterene", ["Psychic"], "Magic Bounce", "Leftovers"),
                                      att_state={"status": "brn"})),
    ("+2 Attack, super effective", dict(att=S("Kingambit", ["Iron Head"], "Supreme Overlord", "Leftovers"),
                                        def_=S("Clefable", ["Moonblast"], "Magic Guard", "Leftovers"),
                                        att_state={"boosts": {"atk": 2}})),
    ("Reflect", dict(att=S("Great Tusk", ["Headlong Rush"], "Protosynthesis", "Leftovers"),
                     def_=S("Ting-Lu", ["Ruination"], "Vessel of Ruin", "Leftovers"),
                     def_side=["reflect"])),
    ("4x into Multiscale", dict(att=S("Iron Bundle", ["Ice Beam"], "Quark Drive", "Choice Specs"),
                                def_=S("Dragonite", ["Extreme Speed"], "Multiscale", "Heavy-Duty Boots"))),
    ("Technician", dict(att=S("Scizor", ["Bullet Punch"], "Technician", "Life Orb"),
                        def_=S("Iron Valiant", ["Moonblast"], "Quark Drive", "Leftovers"))),
    ("Huge Power, Choice Band", dict(att=S("Azumarill", ["Aqua Jet"], "Huge Power", "Choice Band"),
                                     def_=S("Garchomp", ["Earthquake"], "Rough Skin", "Leftovers"))),
    ("Adaptability into Eviolite", dict(att=S("Porygon-Z", ["Tri Attack"], "Adaptability", "Choice Specs"),
                                        def_=S("Chansey", ["Seismic Toss"], "Natural Cure", "Eviolite"))),
    ("Knock Off on an item holder", dict(att=S("Weavile", ["Knock Off"], "Pressure", "Choice Band"),
                                         def_=S("Gholdengo", ["Make It Rain"], "Good as Gold", "Leftovers"))),
    ("Body Press at +2 Def", dict(att=S("Corviknight", ["Body Press"], "Pressure", "Leftovers"),
                                  def_=S("Ting-Lu", ["Ruination"], "Vessel of Ruin", "Leftovers"),
                                  att_state={"boosts": {"def": 2}})),
    ("Foul Play", dict(att=S("Grimmsnarl", ["Foul Play"], "Prankster", "Light Clay"),
                       def_=S("Great Tusk", ["Headlong Rush"], "Protosynthesis", "Leftovers"))),
    ("Psyshock into Blissey", dict(att=S("Alakazam", ["Psyshock"], "Magic Guard", "Life Orb"),
                                   def_=S("Blissey", ["Seismic Toss"], "Natural Cure", "Heavy-Duty Boots"))),
    ("Guts Facade while burned", dict(att=S("Ursaluna", ["Facade"], "Guts", "Flame Orb"),
                                      def_=S("Toxapex", ["Surf"], "Regenerator", "Black Sludge"),
                                      att_state={"status": "brn"})),
    ("Earthquake in Grassy Terrain", dict(att=S("Great Tusk", ["Earthquake"], "Protosynthesis", "Leftovers"),
                                          def_=S("Kingambit", ["Kowtow Cleave"], "Supreme Overlord", "Leftovers"),
                                          terrain="grassyterrain")),
    ("Assault Vest", dict(att=S("Iron Valiant", ["Moonblast"], "Quark Drive", "Choice Specs"),
                          def_=S("Iron Hands", ["Drain Punch"], "Quark Drive", "Assault Vest"))),
    ("Eruption in sun at full HP", dict(att=S("Typhlosion-Hisui", ["Eruption"], "Frisk", "Choice Scarf"),
                                        def_=S("Corviknight", ["Body Press"], "Pressure", "Leftovers"),
                                        weather="sunnyday")),
    ("Sheer Force + Life Orb", dict(att=S("Nidoking", ["Earth Power"], "Sheer Force", "Life Orb"),
                                    def_=S("Heatran", ["Magma Storm"], "Flame Body", "Leftovers"))),
    ("Freeze-Dry into Water", dict(att=S("Iron Bundle", ["Freeze-Dry"], "Quark Drive", "Choice Specs"),
                                   def_=S("Toxapex", ["Surf"], "Regenerator", "Black Sludge"))),
    ("Ogerpon mask, Ivy Cudgel", dict(att=S("Ogerpon-Wellspring", ["Ivy Cudgel"], "Water Absorb", "Wellspring Mask", tera="Water"),
                                      def_=S("Great Tusk", ["Headlong Rush"], "Protosynthesis", "Leftovers"))),
    ("Solid Rock", dict(att=S("Pelipper", ["Hydro Pump"], "Drizzle", "Damp Rock"),
                        def_=S("Rhyperior", ["Earthquake"], "Solid Rock", "Weakness Policy"))),
    ("Fluffy against contact", dict(att=S("Great Tusk", ["Close Combat"], "Protosynthesis", "Leftovers"),
                                    def_=S("Bewear", ["Double-Edge"], "Fluffy", "Leftovers"))),
    ("Tinted Lens", dict(att=S("Venomoth", ["Bug Buzz"], "Tinted Lens", "Life Orb"),
                         def_=S("Heatran", ["Magma Storm"], "Flash Fire", "Leftovers"))),
    ("sandstorm Rock SpD boost", dict(att=S("Pelipper", ["Hydro Pump"], "Drizzle", "Damp Rock"),
                                      def_=S("Tyranitar", ["Stone Edge"], "Sand Stream", "Leftovers"),
                                      weather="sandstorm")),
    ("Electric Terrain boost", dict(att=S("Iron Hands", ["Wild Charge"], "Quark Drive", "Assault Vest"),
                                    def_=S("Corviknight", ["Body Press"], "Pressure", "Leftovers"),
                                    terrain="electricterrain")),
    ("Tera Blast with Tera", dict(att=S("Dragonite", ["Tera Blast"], "Multiscale", "Heavy-Duty Boots", tera="Normal"),
                                  def_=S("Great Tusk", ["Headlong Rush"], "Protosynthesis", "Leftovers"),
                                  att_state={"tera": True})),
    ("Hex into a statused target", dict(att=S("Gholdengo", ["Shadow Ball"], "Good as Gold", "Choice Specs"),
                                        def_=S("Hatterene", ["Psychic"], "Magic Bounce", "Leftovers"),
                                        def_state={"status": "par"})),
]


def engine_mon(d, move):
    return pe.Pokemon(
        id=d["species"], level=d["level"],
        types=tuple((d["base_types"] + ["typeless"])[:2]), base_types=tuple((d["base_types"] + ["typeless"])[:2]),
        hp=d["hp"], maxhp=d["maxhp"], ability=d["ability"], item=d["item"],
        attack=d["stats"]["atk"], defense=d["stats"]["def"], special_attack=d["stats"]["spa"],
        special_defense=d["stats"]["spd"], speed=d["stats"]["spe"], status=STATUS[d["status"]],
        weight_kg=float(d["weight_kg"]), moves=[pe.Move(id=move, pp=16)],
        terastallized=d["terastallized"], tera_type=d["tera_type"] or "typeless")


def engine_damage(case, res):
    move = to_id_str(case["att"]["moves"][0])
    s1 = dict(pokemon=[engine_mon(res["att"], move)])
    s2 = dict(pokemon=[engine_mon(res["def"], "splash")])
    for side, st in ((s1, case.get("att_state")), (s2, case.get("def_state"))):
        for k, v in ((st or {}).get("boosts") or {}).items():
            side[BOOST_FIELDS[k]] = v
    conds = {sc: 5 for sc in case.get("def_side", [])}
    if "lightscreen" in conds:
        conds["light_screen"] = conds.pop("lightscreen")
    state = pe.State(side_one=pe.Side(**s1), side_two=pe.Side(**s2, side_conditions=pe.SideConditions(**conds)),
                     weather=WEATHER[res["weather"]], weather_turns_remaining=5,
                     terrain=res["terrain"] or "none", terrain_turns_remaining=5)
    rolls, _ = pe.calculate_damage(state, move, "splash", True)
    return rolls


def tera_check():
    def mon(sid, types, moves, tera, terad=False):
        return pe.Pokemon(id=sid, level=80, types=types, base_types=types, hp=300, maxhp=300, attack=250,
                          defense=200, special_attack=200, special_defense=200, speed=200,
                          moves=[pe.Move(id=m, pp=16) for m in moves], tera_type=tera, terastallized=terad)

    out = {}
    for terad in (False, True):
        s1 = pe.Side(pokemon=[mon("garchomp", ("dragon", "ground"), ["earthquake", "swordsdance"], "steel"),
                              mon("gholdengo", ("steel", "ghost"), ["makeitrain", "recover"], "fairy", terad)])
        s2 = pe.Side(pokemon=[mon("heatran", ("fire", "steel"), ["magmastorm", "protect"], "grass")])
        r = pe.monte_carlo_tree_search(pe.State(side_one=s1, side_two=s2), iterations=20000)
        out[terad] = ({x.move_choice for x in r.side_one}, {x.move_choice for x in r.side_two})
    ok = (any(o.endswith("-tera") for o in out[False][0]) and not any(o.endswith("-tera") for o in out[True][0])
          and any(o.endswith("-tera") for o in out[True][1]))
    print(f"tera options, side can tera:     {sorted(out[False][0])}")
    print(f"tera options, side already used: {sorted(out[True][0])} (other side: {sorted(out[True][1])})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showdown", default=str(SHOWDOWN))
    args = ap.parse_args()
    print(f"poke-engine {importlib.metadata.version('poke-engine')}")
    tera_ok = tera_check()
    print("tera check:", "OK" if tera_ok else "FAILED (not the gen9 + terastallization build?)")

    cases = [{("def" if k == "def_" else k): v for k, v in c.items()} for _, c in CASES]
    proc = subprocess.run(["node", str(ROOT / "tools" / "showdown_damage.js"), args.showdown],
                          input=json.dumps(cases), capture_output=True, text=True, check=True)
    results = json.loads(proc.stdout)
    print(f"\n{'case':34s} {'Showdown max/min (crit)':>24s} {'engine max (crit)':>18s}  diff")
    n_exact = n_close = 0
    for (name, _), case, res in zip(CASES, cases, results):
        eng = engine_damage(case, res)
        smax, smin, scrit = res["max"], res["min"], res["crit"]
        emax, ecrit = (eng + [0, 0])[:2]
        d = (emax - smax) if smax is not None else None
        rel = abs(d) / max(smax, 1) if d is not None else 1.0
        exact = d == 0 and ecrit == scrit
        close = d is not None and rel <= 0.01 and abs(ecrit - (scrit or 0)) <= max(1, 0.01 * (scrit or 0))
        n_exact += exact
        n_close += close
        flag = "exact" if exact else ("within 1%" if close else "DIFFERENT")
        print(f"{name:34s} {f'{smax}/{smin} ({scrit})':>24s} {f'{emax} ({ecrit})':>18s}  "
              f"{d if d is not None else '-':>4} {flag}")
    print(f"\n{len(CASES)} cases: {n_exact} exact, {n_close} within 1% (incl. exact), "
          f"{len(CASES) - n_close} different")


if __name__ == "__main__":
    main()
