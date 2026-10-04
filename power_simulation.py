# -*- coding: utf-8 -*-
"""
power_simulation.py — Monte Carlo power analysis.

Simple idea: generate many (default 1000) independent "experiments"
with the same planted effects, run the same analysis in each, and calculate
the proportion of replications in which the hypothesis is confirmed.

How to run:
    python power_simulation.py                  # 1000 replications (~1–3 minutes)
    python power_simulation.py --n-sim 200      # quick check
Author signature: a1d0s_3l1u45
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import sim_core as sc
import analysis_core as ac

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    parser = argparse.ArgumentParser(description="Power simulation (Monte Carlo)")
    parser.add_argument("--n-sim", type=int, default=1000, help="number of experiment replications")
    parser.add_argument("--seed0", type=int, default=100000, help="initial seed")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []
    failed = 0
    start = time.time()
    for i in range(args.n_sim):
        try:
            st, tk, _ = sc.generate_dataset(args.seed0 + i)
            res = ac.run_confirmatory(st, tk, light=True)  # light=True: skip slow supplementary tests
            ht = res["holm_table"]
            rec = {}
            for _, row in ht.iterrows():
                key = row["hypothesis"].split(" ")[0]  # H1a, H2a, H2b, H3
                rec[f"{key}_raw"] = bool((row["p_raw"] < ac.ALPHA) and row["direction_as_hypothesized"])
                rec[f"{key}_holm"] = bool(row["confirmed"])
            rec["H1b_equiv"] = bool(res["h1b"]["equivalent"])
            records.append(rec)
        except Exception as e:  # a single failed replication should not abort the whole calculation
            failed += 1
            if failed <= 3:
                print(f"Replication {i}: error {type(e).__name__}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  completed {i + 1}/{args.n_sim} ({time.time() - start:.0f} s)")

    df = pd.DataFrame(records)
    n_ok = len(df)
    if n_ok == 0:
        print("Failed to complete any replications.")
        return

    labels = {"H1a": "H1a: EG faster on explicit defects",
              "H2a": "H2a: EL lower in EG",
              "H2b": "H2b: GL lower in EG",
              "H3": "H3: Transfer higher in CG"}
    rows = []
    for key, text in labels.items():
        rows.append({"Hypothesis": text,
                     "Unadjusted power": df[f"{key}_raw"].mean(),
                     "Holm-adjusted power": df[f"{key}_holm"].mean()})
    rows.append({"Hypothesis": "H1b: Equivalence (TOST, ±0.25) confirmed",
                 "Unadjusted power": df["H1b_equiv"].mean(),
                 "Holm-adjusted power": np.nan})
    table = pd.DataFrame(rows)
    table["Unadjusted power"] = table["Unadjusted power"].round(3)
    table["Holm-adjusted power"] = table["Holm-adjusted power"].round(3)

    # Proportion of replications where all 4 directional hypotheses were confirmed simultaneously
    all4 = (df[["H1a_holm", "H2a_holm", "H2b_holm", "H3_holm"]].all(axis=1)).mean()

    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 60)
    print(f"\nSuccessful replications: {n_ok}, failed: {failed}")
    print(table.to_string(index=False))
    print(f"\nProportion of replications where ALL four hypotheses were confirmed (with Holm): {all4:.3f}")
    print("Standard error of power estimate ≈ "
          f"{np.sqrt(0.25 / n_ok):.3f} (worst case, p = 0.5).")

    table.to_csv(os.path.join(OUT_DIR, "power_results.csv"), index=False, encoding="utf-8-sig")
    print(f"\nTable saved: {os.path.join(OUT_DIR, 'power_results.csv')}")


if __name__ == "__main__":
    main()
