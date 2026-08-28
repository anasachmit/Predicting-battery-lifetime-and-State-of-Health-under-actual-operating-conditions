# -*- coding: utf-8 -*-
"""Genere le pool fige de laboratoires synthetiques du BatteryPFN.

    python scripts/make_pool.py --smoke          # 5 000 labos, validation format
    python scripts/make_pool.py                  # 550 000 labos, pool complet

Pool : 500 000 laboratoires d'entrainement (seed A) + 50 000 de validation
(seed B, DISJOINTE). Shards de 50 000, ecrits en parquet, un manifest JSON.

Pourquoi un pool FIGE plutot qu'une generation a la volee
---------------------------------------------------------
Un PFN se pre-entraine normalement sur un flux infini, ce qui rend le
surapprentissage impossible. Figer le pool le rend possible - et c'est un choix
assume : il rend l'entrainement reproductible, verifiable et rejouable, ce qui
compte davantage ici que l'acces a une variete infinie. La contrepartie est
qu'il faut SURVEILLER l'epuisement du pool, d'ou la seed de validation
disjointe et le suivi separe de la calibration sur les deux pools (cf. F.1
dans VERDICT.md).

Robustesse d'execution
----------------------
Chaque shard est ecrit puis verifie independamment, et le script est
RESUMABLE : relance-le, il saute les shards deja presents et valides. C'est
necessaire ici parce que Smart App Control peut bloquer une DLL au premier
acces et tuer un run long ; perdre un shard ne doit pas coûter le pool entier.

Format d'un shard
-----------------
Deux tables par shard, pour ne pas dupliquer les theta a chaque point :

  `labs`   une ligne par CELLULE (6 par laboratoire) : theta vrais, conditions,
           et les statistiques de la trajectoire.
  `traj`   une ligne par POINT observe : (lab_id, cell_id, cycle, soh).

C'est le format que consommera le chargeur d'entrainement, qui sous-echantillonne
les trajectoires a la longueur de contexte du reseau.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pfn.prior import DEFAULT                       # noqa: E402
from src.pfn.generate import simulate_lab               # noqa: E402
from src.pfn.prior import draw_lab                      # noqa: E402

POOL_DIR = ROOT / "data" / "pfn_pool"

# Seeds DISJOINTES, et separees par un ecart enorme pour qu'aucun recouvrement
# de sous-sequence ne soit possible entre les deux flux.
SEED_TRAIN = 20260101
SEED_VALID = 77770101

N_TRAIN = 500_000
N_VALID = 50_000
SHARD = 50_000

# Decalage d'identifiant par split. Sans lui, `lab_id = shard * SHARD + k`
# redemarre a 0 pour la validation et COLLISIONNE avec les identifiants
# d'entrainement : les theta restent disjoints, mais tout code qui
# concatenerait les deux tables melangerait silencieusement les laboratoires.
# Le decalage rend `lab_id` globalement unique et rend la faute impossible.
OFFSET_LAB_ID = {"train": 0, "valid": 10_000_000}

# Sous-echantillonnage : une trajectoire complete peut faire 12 000 points, ce
# qui rendrait le pool inutilisablement gros (550k x 6 x ~3000 points). Le
# reseau ne verra de toute facon qu'un contexte borne. On conserve donc
# MAX_POINTS points par trajectoire, choisis sur une grille GEOMETRIQUE : la
# trajectoire bouge vite au debut et lentement ensuite, un pas uniforme
# gaspillerait la resolution la ou elle ne sert pas.
MAX_POINTS = 128


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "inconnu"


def git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return None


def _sous_echantillonne(n, y):
    """Grille geometrique de MAX_POINTS indices, premier et dernier conserves."""
    if len(n) <= MAX_POINTS:
        return n, y
    idx = np.unique(np.round(
        np.geomspace(1, len(n), MAX_POINTS)).astype(int) - 1)
    idx = np.clip(idx, 0, len(n) - 1)
    idx[-1] = len(n) - 1
    return n[idx], y[idx]


def genere_shard(args):
    """Un shard complet. Execute dans un process separe.

    `pool_dir` est passe EXPLICITEMENT dans la tache : un sous-process reimporte
    le module et ne verrait pas une reaffectation du global faite dans
    `__main__`. Sans cela, un run `--smoke` ecrirait ses shards dans le
    repertoire du vrai pool.
    """
    shard_id, seed, n_labs, split, pool_dir = args
    pool_dir = Path(pool_dir)
    rng = np.random.default_rng([seed, shard_id])
    lignes_lab, lignes_traj = [], []
    for k in range(n_labs):
        lab_id = OFFSET_LAB_ID[split] + shard_id * SHARD + k
        lab = simulate_lab(draw_lab(rng, DEFAULT), rng, DEFAULT)
        th = lab["theta"]
        for ci, tr in enumerate(lab["trajectoires"]):
            t = tr["theta"]
            n, y = _sous_echantillonne(tr["n"], tr["y"])
            lignes_lab.append((
                lab_id, ci, lab["structure_nom"],
                lab["ea_25_35"], lab["ea_35_45"], lab["ea_45_55"],
                th["loss_ref"], th["p"], th["soh_k"], th["g"], th["tau_frac"],
                th["w2"], th["sigma_meas"], th["sigma_cell"], th["sigma_modele"],
                t["T"], t["C"], t["b"], t["s0"],
                float(tr["n_max"]), len(n), float(tr["y"][-1])))
            lignes_traj.append(pd.DataFrame(
                {"lab_id": np.int32(lab_id), "cell_id": np.int8(ci),
                 "cycle": n.astype(np.int32), "soh": y.astype(np.float32)}))

    labs = pd.DataFrame(lignes_lab, columns=[
        "lab_id", "cell_id", "structure", "ea_25_35", "ea_35_45", "ea_45_55",
        "loss_ref", "p", "soh_k", "g", "tau_frac", "w2", "sigma_meas",
        "sigma_cell", "sigma_modele", "T", "C", "b", "s0",
        "n_max", "n_points", "soh_final"])
    traj = pd.concat(lignes_traj, ignore_index=True)

    d = pool_dir / split
    d.mkdir(parents=True, exist_ok=True)
    f_lab = d / f"labs_{shard_id:03d}.parquet"
    f_traj = d / f"traj_{shard_id:03d}.parquet"
    labs.to_parquet(f_lab, index=False)
    traj.to_parquet(f_traj, index=False)

    return dict(
        shard=shard_id, split=split, n_labs=int(n_labs),
        n_cellules=int(len(labs)), n_points=int(len(traj)),
        fichier_labs=f_lab.name, fichier_traj=f_traj.name,
        sha256_labs=_sha(f_lab), sha256_traj=_sha(f_traj),
        octets=int(f_lab.stat().st_size + f_traj.stat().st_size),
        structures={k: int(v) for k, v in
                    labs.groupby("lab_id").structure.first()
                        .value_counts().items()},
        p_q=[float(x) for x in np.percentile(
            labs.groupby("lab_id").p.first(), [5, 25, 50, 75, 95])])


def _sha(path, bloc=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bloc), b""):
            h.update(chunk)
    return h.hexdigest()


def _shard_valide(split, shard_id, pool_dir):
    d = Path(pool_dir) / split
    fl, ft = d / f"labs_{shard_id:03d}.parquet", d / f"traj_{shard_id:03d}.parquet"
    return fl.exists() and ft.exists() and fl.stat().st_size > 0 \
        and ft.stat().st_size > 0


def run(n_train, n_valid, shard, workers, pool_dir, force=False):
    t0 = time.time()
    taches = []
    for split, seed, total in (("train", SEED_TRAIN, n_train),
                               ("valid", SEED_VALID, n_valid)):
        nsh = (total + shard - 1) // shard
        for s in range(nsh):
            n = min(shard, total - s * shard)
            if not force and _shard_valide(split, s, pool_dir):
                print(f"  [saut] {split} shard {s:03d} deja present")
                continue
            taches.append((s, seed, n, split, str(pool_dir)))

    infos = []
    if taches:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(genere_shard, t): t for t in taches}
            for i, fut in enumerate(as_completed(futs), 1):
                info = fut.result()
                infos.append(info)
                print(f"  [{i}/{len(taches)}] {info['split']} shard "
                      f"{info['shard']:03d} : {info['n_labs']} labos, "
                      f"{info['n_points']:,} points, "
                      f"{info['octets']/1e6:.1f} Mo", flush=True)

    manifest = ecrire_manifest(infos, n_train, n_valid, shard, pool_dir,
                               duree_s=time.time() - t0)
    return manifest


def ecrire_manifest(infos, n_train, n_valid, shard, pool_dir, duree_s):
    from dataclasses import asdict
    pool_dir = Path(pool_dir)
    chemin = pool_dir / "manifest.json"
    anciens = []
    if chemin.exists():
        anciens = json.loads(chemin.read_text(encoding="utf-8")).get("shards", [])
    connus = {(i["split"], i["shard"]) for i in infos}
    shards = sorted(infos + [a for a in anciens
                             if (a["split"], a["shard"]) not in connus],
                    key=lambda s: (s["split"], s["shard"]))

    agg = {}
    p_all = []
    for s in shards:
        for k, v in s["structures"].items():
            agg[k] = agg.get(k, 0) + v
        p_all.append(s["p_q"])
    tot = sum(agg.values()) or 1
    manifest = dict(
        cree_le=time.strftime("%Y-%m-%dT%H:%M:%S"),
        duree_generation_s=round(duree_s, 1),
        commit_git=git_commit(),
        arbre_git_modifie=git_dirty(),
        version_prior="BatteryPFN prior fige 2026-08-26 "
                      "(L_ref recentre, p ~ N(0.55, 0.15) tronquee, "
                      "erreur de modele correlee)",
        seed_train=SEED_TRAIN, seed_valid=SEED_VALID,
        n_train=n_train, n_valid=n_valid, taille_shard=shard,
        max_points_par_trajectoire=MAX_POINTS,
        n_shards=len(shards),
        n_laboratoires=sum(s["n_labs"] for s in shards),
        n_cellules=sum(s["n_cellules"] for s in shards),
        n_points=sum(s["n_points"] for s in shards),
        octets=sum(s["octets"] for s in shards),
        distribution_structures={k: dict(n=v, part=round(v / tot, 4))
                                 for k, v in sorted(agg.items())},
        distribution_p=dict(zip(
            ["q05", "q25", "q50", "q75", "q95"],
            [round(float(np.mean([p[i] for p in p_all])), 4)
             for i in range(5)])) if p_all else {},
        parametres_prior={k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in asdict(DEFAULT).items()},
        shards=shards)
    pool_dir.mkdir(parents=True, exist_ok=True)
    chemin.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return manifest


def verifie(manifest, tol=0.02):
    """Controle du manifest. La distribution des structures est BLOQUANTE.

    Un pool sous-representant la charniere affaiblirait mecaniquement F.3 et on
    ne s'en apercevrait qu'a la fin.
    """
    attendu = dict(arrhenius=0.45, charniere=0.45, quadratique=0.10)
    ok, msgs = True, []
    for nom, part_attendue in attendu.items():
        obs = manifest["distribution_structures"].get(nom, {}).get("part", 0.0)
        bon = abs(obs - part_attendue) <= tol
        ok &= bon
        msgs.append(f"  {'OK   ' if bon else 'ECHEC'} {nom:12s} "
                    f"{obs:.4f} (attendu {part_attendue:.2f} +/- {tol})")
    n_ok = manifest["n_laboratoires"] == manifest["n_train"] + manifest["n_valid"]
    ok &= n_ok
    msgs.append(f"  {'OK   ' if n_ok else 'ECHEC'} nombre de laboratoires "
                f"{manifest['n_laboratoires']:,}")
    inter = set()
    for s in manifest["shards"]:
        cle = (s["split"], s["shard"])
        if cle in inter:
            ok = False
            msgs.append(f"  ECHEC shard duplique {cle}")
        inter.add(cle)
    return ok, msgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="5 000 labos au lieu de 550 000")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        n_tr, n_va, sh = 4_000, 1_000, 1_000
        pool_dir = POOL_DIR.with_name("pfn_pool_smoke")
    else:
        n_tr, n_va, sh = N_TRAIN, N_VALID, SHARD
        pool_dir = POOL_DIR

    print(f"pool -> {pool_dir}")
    print(f"  {n_tr:,} entrainement (seed {SEED_TRAIN}) + "
          f"{n_va:,} validation (seed {SEED_VALID}), shards de {sh:,}")
    print(f"  {a.workers} process")
    m = run(n_tr, n_va, sh, a.workers, pool_dir, force=a.force)

    print(f"\nmanifest : {POOL_DIR / 'manifest.json'}")
    print(f"  commit git      : {m['commit_git'][:12]} "
          f"(arbre modifie : {m['arbre_git_modifie']})")
    print(f"  laboratoires    : {m['n_laboratoires']:,}")
    print(f"  cellules        : {m['n_cellules']:,}")
    print(f"  points          : {m['n_points']:,}")
    print(f"  taille          : {m['octets']/1e9:.2f} Go")
    print(f"  duree           : {m['duree_generation_s']:.0f} s")
    print(f"  structures      : "
          + ", ".join(f"{k} {v['part']:.4f}"
                      for k, v in m["distribution_structures"].items()))
    print(f"  p (quantiles)   : {m['distribution_p']}")
    ok, msgs = verifie(m)
    print("\ncontrole du manifest :")
    print("\n".join(msgs))
    print(f"\n{'POOL VALIDE' if ok else 'POOL INVALIDE - NE PAS ENTRAINER DESSUS'}")
    sys.exit(0 if ok else 1)
