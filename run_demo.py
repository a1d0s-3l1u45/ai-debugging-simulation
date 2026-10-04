# -*- coding: utf-8 -*-
"""
run_demo.py — MAIN SCRIPT: generates one synthetic dataset,
saves it to CSV, performs analysis following the revised specification, and writes report and figures.

How to run:
    python run_demo.py              # default seed = 2026
    python run_demo.py --seed 7     # another dataset

Results will appear in the output/ directory next to the script.
Author signature: a1d0s_3l1u45
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to files without window display (works on headless servers)
import matplotlib.pyplot as plt
from statsmodels.duration.survfunc import SurvfuncRight

import sim_core as sc
import analysis_core as ac

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def fmt_p(p):
    """Formatted p-value output (APA: p < .001, without leading zero)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".")


def descriptives(students, tasks):
    """Descriptive statistics table by group (mean and SD; for scales — median and IQR)."""
    rows = []

    def add(label, col, df=students, kind="mean"):
        row = {"Metric": label}
        for g in ("CG", "EG"):
            x = df.loc[df.group == g, col].dropna()
            if kind == "mean":
                row[g] = f"{x.mean():.2f} ({x.std(ddof=1):.2f})"
            else:
                q1, q3 = np.percentile(x, [25, 75])
                row[g] = f"{x.median():.2f} [{q1:.2f}; {q3:.2f}]"
        rows.append(row)

    add("Pre-test, M (SD)", "pre_score")
    add("Explicit tasks solved (0–2), M (SD)", "SR_expl_n")
    add("Hidden tasks solved (0–2), M (SD)", "SR_hid_n")
    add("Total pytest runs, M (SD)", "attempts_total")
    add("IL, Mdn [Q1; Q3]", "IL", kind="median")
    add("EL, Mdn [Q1; Q3]", "EL", kind="median")
    add("GL, Mdn [Q1; Q3]", "GL", kind="median")
    add("Transfer (0–10), M (SD)", "transfer_score")
    return pd.DataFrame(rows)


def make_figures(students, tasks):
    """Three figures for the Results section."""
    colors = {"CG": "#4C72B0", "EG": "#DD8452"}
    names = {"CG": "CG", "EG": "EG"}

    # --- Figure 1: cumulative proportion of solvers on explicit tasks (Kaplan–Meier) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, t in zip(axes, ("T1", "T2")):
        for g in ("CG", "EG"):
            d = tasks[(tasks.task == t) & (tasks.group == g) & (tasks.attempted == 1)]
            sf = SurvfuncRight(d["time_min"].values, d["solved"].values)
            times = np.concatenate([[0.0], sf.surv_times])
            solved_share = np.concatenate([[0.0], 1.0 - sf.surv_prob])
            ax.step(times, solved_share, where="post", color=colors[g], label=names[g], lw=2)
        ax.set_title(f"Task {t} (explicit defect)")
        ax.set_xlabel("Time, min")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Proportion solved (Kaplan–Meier estimate)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_km_explicit.png"), dpi=150)
    plt.close(fig)

    # --- Figure 2: cognitive load by subscale ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    rng = np.random.default_rng(0)  # jitter for points to avoid overplotting
    for ax, sc_name in zip(axes, ("IL", "EL", "GL")):
        data = [students.loc[students.group == g, sc_name].values for g in ("CG", "EG")]
        ax.boxplot(data, tick_labels=["CG", "EG"], widths=0.5, showfliers=False)
        for k, arr in enumerate(data, start=1):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(arr)), arr, s=18, alpha=0.6)
        ax.set_title(sc_name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Item mean (0–10)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_cognitive_load.png"), dpi=150)
    plt.close(fig)

    # --- Figure 3: Transfer vs Pre-test ---
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for g in ("CG", "EG"):
        d = students[students.group == g]
        ax.scatter(d["pre_score"], d["transfer_score"], color=colors[g], label=names[g], alpha=0.8)
        slope, intercept = np.polyfit(d["pre_score"], d["transfer_score"], 1)
        xs = np.linspace(d["pre_score"].min(), d["pre_score"].max(), 20)
        ax.plot(xs, intercept + slope * xs, color=colors[g])
    ax.set_xlabel("Pre-test (0–20)")
    ax.set_ylabel("Transfer test (0–10)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_transfer_vs_pre.png"), dpi=150)
    plt.close(fig)


def build_report(seed, meta, students, tasks, res, desc):
    """Assembles text report (Markdown) with all results."""
    h1a, h1b, h2, h3, holm = res["h1a"], res["h1b"], res["h2"], res["h3"], res["holm_table"]
    bal = ac.pretest_balance(students)
    alphas = {
        "IL": ac.cronbach_alpha(students[["IL1", "IL2"]]),
        "EL": ac.cronbach_alpha(students[["EL1", "EL2", "EL3"]]),
        "GL": ac.cronbach_alpha(students[["GL1", "GL2", "GL3"]]),
    }
    L = []
    L.append("# Report on SYNTHETIC dataset (educational example)\n")
    L.append("> **WARNING:** data were generated by software and not obtained from an experiment. "
             "Effects were planted by the simulation author. Results must not be presented as empirical.\n")
    L.append(f"Seed = {seed}. Protocol violators (EG): {', '.join(map(str, meta['protocol_violators']))}. "
             f"CG student who solved no tasks: {meta['zero_solver_id']}.\n")

    L.append("## 1. Group balance and descriptive statistics\n")
    L.append(f"Pre-test: CG M = {bal['mean_cg']:.2f}, EG M = {bal['mean_eg']:.2f}; "
             f"paired t-test p {fmt_p(bal['paired_t_p'])}.\n")
    L.append("```\n" + desc.to_string(index=False) + "\n```\n")
    L.append("Subscale reliability (Cronbach's α; 2–3 items, hence coarse estimates): "
             + ", ".join(f"{k} = {v:.2f}" for k, v in alphas.items()) + ".\n")

    L.append("## 2. Confirmatory hypotheses (two-sided tests, α = .05, Holm correction)\n")
    t = holm.copy()
    t["p_raw"] = t["p_raw"].map(lambda v: f"{v:.4f}")
    t["p_holm"] = t["p_holm"].map(lambda v: f"{v:.4f}")
    L.append("```\n" + t.to_string(index=False) + "\n```\n")

    L.append("### H1a — solution time for explicit defects (Cox regression, robust SE clustered by student)\n")
    L.append(f"HR (EG/CG) = {h1a['hr']:.2f}, 95% CI [{h1a['hr_ci'][0]:.2f}; {h1a['hr_ci'][1]:.2f}], "
             f"p {fmt_p(h1a['p'])}; observations {h1a['n_obs']}, solved {h1a['n_events']}. "
             "HR > 1 means EG solves faster.\n")
    pt = h1a["per_task"].copy()
    L.append("Kaplan–Meier medians (min) and log-rank test by task:\n")
    L.append("```\n" + pt.round(3).to_string(index=False) + "\n```\n")
    sa = h1a["secondary_ancova"]
    L.append(f"Secondary analysis (ANCOVA on mean ln(time) of solved explicit tasks, n = {sa['n']}): "
             f"b(EG) = {sa['coef_eg']:.2f}, p {fmt_p(sa['p'])}, partial η² = {sa['eta2_p']:.2f}; "
             f"Shapiro–Wilk on residuals p {fmt_p(sa['shapiro_resid_p'])}, Levene p {fmt_p(sa['levene_p'])}.\n")

    L.append("### H1b — equivalence on hidden defects (TOST, bounds ±0.25)\n")
    L.append(f"Proportion of solved hidden tasks: CG = {h1b['mean_cg']:.3f}, EG = {h1b['mean_eg']:.3f}; "
             f"difference = {h1b['diff']:.3f}, 90% CI [{h1b['ci90'][0]:.3f}; {h1b['ci90'][1]:.3f}]; "
             f"p(TOST) {fmt_p(h1b['p_tost'])} -> "
             f"{'equivalence confirmed' if h1b['equivalent'] else 'equivalence NOT confirmed'}.\n")
    L.append(f"Supplementary: GEE logistic model across tasks OR(EG) = {h1b['gee_or']:.2f}, "
             f"p {fmt_p(h1b['gee_p'])}; permutation test on number of solved tasks p {fmt_p(h1b['perm_p'])}.\n")

    L.append("### H2 — cognitive load (paired Wilcoxon test; sensitivity analysis — Mann–Whitney)\n")
    rows = []
    for k, v in h2.items():
        rows.append({"Scale": k, "Mdn CG": v["median_CG"], "Mdn EG": v["median_EG"],
                     "Wilcoxon p": round(v["wilcoxon_p"], 4), "r (Wilcox.)": round(v["r_wilcoxon"], 2),
                     "Mann—Whitney p": round(v["mw_p"], 4), "r (M—W)": round(v["r_mw"], 2)})
    L.append("```\n" + pd.DataFrame(rows).round(2).to_string(index=False) + "\n```\n")
    L.append("r = Z / √N, where N = 40 (number of observations); Z obtained from two-sided p. IL is exploratory analysis "
             "(not part of confirmatory hypotheses).\n")

    L.append("### H3 — Transfer test (ANCOVA, Pre-test covariate)\n")
    L.append(f"Adjusted means: CG = {h3['adj_mean_cg']:.2f}, EG = {h3['adj_mean_eg']:.2f}; "
             f"b(EG) = {h3['coef_eg']:.2f} (SE = {h3['se']:.2f}), t = {h3['t']:.2f}, p {fmt_p(h3['p'])}, "
             f"partial η² = {h3['eta2_p']:.3f}; Shapiro–Wilk on residuals p {fmt_p(h3['shapiro_resid_p'])}, "
             f"Levene p {fmt_p(h3['levene_p'])}.\n")

    L.append("## 3. Important notes for students\n")
    L.append("- Effects are planted by the simulation; \"confirming\" hypotheses here proves nothing about real AI.\n"
             "- With n = 20 per group, power for moderate effects is modest (see power_simulation.py): "
             "some hypotheses fail to confirm even with a true underlying effect.\n"
             "- Non-significant result after Holm correction despite being significant before it is common; "
             "report it honestly in Discussion and Limitations.\n")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description="Synthetic data generation and analysis")
    parser.add_argument("--seed", type=int, default=2026, help="generator seed (default 2026)")
    args = parser.parse_args()

    try:
        os.makedirs(OUT_DIR, exist_ok=True)
    except OSError as e:
        print(f"Failed to create directory {OUT_DIR}: {e}")
        sys.exit(1)

    # 1. Generation
    students, tasks, meta = sc.generate_dataset(args.seed)

    # 2. Save data (utf-8-sig — for Excel compatibility)
    students.to_csv(os.path.join(OUT_DIR, "students_wide.csv"), index=False, encoding="utf-8-sig")
    tasks.to_csv(os.path.join(OUT_DIR, "tasks_long.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(sc.DATA_DICTIONARY, columns=["variable", "description", "scale"]).to_csv(
        os.path.join(OUT_DIR, "data_dictionary.csv"), index=False, encoding="utf-8-sig")

    # 3. Analysis
    res = ac.run_confirmatory(students, tasks, light=False)
    desc = descriptives(students, tasks)

    # 4. Report and figures
    report = build_report(args.seed, meta, students, tasks, res, desc)
    with open(os.path.join(OUT_DIR, "results_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    make_figures(students, tasks)

    print(report)
    print(f"\nDone. Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
