# -*- coding: utf-8 -*-
"""Entrainement du BatteryPFN sur le pool fige.

    python scripts/train_pfn.py                    # lance ou REPREND
    python scripts/train_pfn.py --labos 500000     # budget en laboratoires vus

Reprenable
----------
Un checkpoint complet (poids, etat de l'optimiseur, compteur de pas, etat du
generateur aleatoire) est ecrit regulierement. Relancer la commande reprend
exactement ou l'entrainement s'etait arrete. C'est necessaire pour deux
raisons : Smart App Control peut bloquer une DLL au premier acces et tuer le
process, et on veut pouvoir prolonger le run le 28 si le temps le permet.

Surveillance - la regle d'arret
-------------------------------
La loss et la COUVERTURE DES QUANTILES sont mesurees separement sur le pool
d'entrainement et sur celui de validation (seeds disjointes). Avec un pool fige,
le surapprentissage redevient possible et se manifeste sur la calibration AVANT
de se voir sur la loss. Si la couverture tient sur l'entrainement et derive sur
la validation, le pool est epuise : le script le signale explicitement.

Un PFN mal calibre est pire que pas de PFN - c'est plus grave qu'une loss elevee.

F.3 aux checkpoints
-------------------
F.3 est evalue a plusieurs jalons (100k, 250k laboratoires vus, puis a la fin),
pas seulement a l'arrivee. Avec un budget d'environ une epoque, un F.3 final
plat serait ininterpretable : impossible de distinguer "le mecanisme ne marche
pas" de "le reseau n'a pas assez vu". La COURBE tranche.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS                              # noqa: E402
from src.pfn.data import GenerateurBatches                  # noqa: E402
from src.pfn.model import (BatteryPFN, perte_pinball,       # noqa: E402
                           compte_parametres)
from src.pfn.evaluate import (evalue_f3, resume_f3,         # noqa: E402
                              evalue_calibration, ecart_calibration)

SORTIE = RESULTS / "pfn" / "entrainement"
CKPT = SORTIE / "checkpoint.pt"
JOURNAL = SORTIE / "journal.jsonl"

N_CTX = 32          # mesure : 24 degrade SOH_k de 31 %, 32 est a +0.8 %
N_QUERY = 48
BATCH = 64
LR = 3e-4
WARMUP = 200

# Jalons F.3, en laboratoires vus.
JALONS_F3 = (100_000, 250_000)


def journalise(**kw):
    SORTIE.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")


def lr_a(pas, base=LR, warmup=WARMUP, total=None):
    if pas < warmup:
        return base * (pas + 1) / warmup
    if total is None:
        return base
    p = min(1.0, (pas - warmup) / max(total - warmup, 1))
    return base * (0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labos", type=int, default=500_000)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--valide-tous", type=int, default=200, help="pas")
    ap.add_argument("--ckpt-tous", type=int, default=200, help="pas")
    ap.add_argument("--repartir-de-zero", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(20260101)
    SORTIE.mkdir(parents=True, exist_ok=True)

    total_pas = a.labos // a.batch
    modele = BatteryPFN()
    opt = torch.optim.AdamW(modele.parameters(), lr=LR, weight_decay=0.01)

    pas0, labos_vus, jalons_faits = 0, 0, []
    if CKPT.exists() and not a.repartir_de_zero:
        etat = torch.load(CKPT, map_location="cpu", weights_only=False)
        modele.load_state_dict(etat["modele"])
        opt.load_state_dict(etat["opt"])
        pas0 = etat["pas"]; labos_vus = etat["labos_vus"]
        jalons_faits = etat.get("jalons_faits", [])
        print(f"reprise au pas {pas0} ({labos_vus:,} laboratoires vus)")

    gen_tr = GenerateurBatches("train", batch=a.batch, n_ctx=a.n_ctx,
                               n_query=N_QUERY, seed=1000 + pas0,
                               reutilisation=400)
    gen_va = GenerateurBatches("valid", batch=a.batch, n_ctx=a.n_ctx,
                               n_query=N_QUERY, seed=7, reutilisation=10_000)
    it_tr, it_va = iter(gen_tr), iter(gen_va)

    print(f"parametres : {compte_parametres(modele):,} | pas vises : {total_pas:,} "
          f"| batch {a.batch} | n_ctx {a.n_ctx} | {a.threads} threads")
    journalise(evenement="depart", pas=pas0, labos_vus=labos_vus,
               total_pas=total_pas, batch=a.batch, n_ctx=a.n_ctx,
               parametres=compte_parametres(modele),
               horodatage=time.strftime("%Y-%m-%dT%H:%M:%S"))

    t0, pertes = time.time(), []
    for pas in range(pas0, total_pas):
        for g in opt.param_groups:
            g["lr"] = lr_a(pas, total=total_pas)
        Xc, Xq, Y, W = next(it_tr)
        q = modele(torch.tensor(Xc), torch.tensor(Xq))
        perte = perte_pinball(q, torch.tensor(Y), modele.quantiles,
                              poids=torch.tensor(W))
        opt.zero_grad(set_to_none=True)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        opt.step()
        pertes.append(perte.detach().item())
        labos_vus += a.batch

        if (pas + 1) % a.valide_tous == 0:
            ctr = evalue_calibration(modele, gen_tr, n_batches=4)
            cva = evalue_calibration(modele, gen_va, n_batches=4)
            etr, eva = ecart_calibration(ctr["couverture"]), \
                ecart_calibration(cva["couverture"])
            derive = eva - etr
            dt = time.time() - t0
            print(f"pas {pas+1:6d} | {labos_vus:8,} labos | "
                  f"perte {np.mean(pertes[-a.valide_tous:]):.4f} | "
                  f"valid {cva['perte']:.4f} | "
                  f"ecart calib train {etr:.4f} valid {eva:.4f} "
                  f"(derive {derive:+.4f}) | {labos_vus/max(dt,1e-9):.0f} lab/s",
                  flush=True)
            journalise(evenement="validation", pas=pas + 1,
                       labos_vus=labos_vus,
                       perte_train=float(np.mean(pertes[-a.valide_tous:])),
                       perte_valid=cva["perte"],
                       couverture_train=ctr["couverture"],
                       couverture_valid=cva["couverture"],
                       ecart_calib_train=etr, ecart_calib_valid=eva,
                       derive_calibration=round(derive, 5),
                       lr=lr_a(pas, total=total_pas),
                       secondes=round(dt, 1))
            # Regle d'arret : la calibration tient sur train mais derive sur
            # valid => le pool fige est epuise. On le SIGNALE, on ne compense pas.
            if etr < 0.02 and derive > 0.03:
                print("\n*** POOL EPUISE : la calibration tient sur "
                      "l'entrainement et derive sur la validation. ***")
                journalise(evenement="pool_epuise", pas=pas + 1,
                           labos_vus=labos_vus, ecart_calib_train=etr,
                           ecart_calib_valid=eva)
                break

        for jal in JALONS_F3:
            if labos_vus >= jal and jal not in jalons_faits:
                jalons_faits.append(jal)
                d = evalue_f3(modele, n_ctx=a.n_ctx, n_labs=250, seed=11)
                r = resume_f3(d)
                d.to_csv(SORTIE / f"f3_{jal}.csv", index=False)
                print(f"  [F.3 @ {jal:,} labos] arrhenius {r['ape_arrhenius']} % | "
                      f"charniere {r['ape_charniere']} % | seuil {r['seuil_F3']:.2f} % | "
                      f"atteint : {r['F3_atteint']}", flush=True)
                journalise(evenement="F3", jalon=jal, labos_vus=labos_vus,
                           pas=pas + 1, **r)

        if (pas + 1) % a.ckpt_tous == 0:
            torch.save(dict(modele=modele.state_dict(), opt=opt.state_dict(),
                            pas=pas + 1, labos_vus=labos_vus,
                            jalons_faits=jalons_faits,
                            n_ctx=a.n_ctx, batch=a.batch), CKPT)

    torch.save(dict(modele=modele.state_dict(), opt=opt.state_dict(),
                    pas=total_pas, labos_vus=labos_vus,
                    jalons_faits=jalons_faits, n_ctx=a.n_ctx, batch=a.batch),
               CKPT)
    d = evalue_f3(modele, n_ctx=a.n_ctx, n_labs=400, seed=11)
    r = resume_f3(d)
    d.to_csv(SORTIE / "f3_final.csv", index=False)
    journalise(evenement="F3", jalon="final", labos_vus=labos_vus,
               pas=total_pas, **r)
    print("\n=== F.3 FINAL ===")
    for k, v in r.items():
        print(f"  {k:24s} {v}")
    print(f"\ncheckpoint : {CKPT}")


if __name__ == "__main__":
    main()
