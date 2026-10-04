# -*- coding: utf-8 -*-
"""
scenarios_simulation.py — FINAL version (v3): operating characteristics of the analysis plan.

What it computes (see article, section 2):
  Scenarios: S0_null (no effects), S1_planted (planted effects), S2_N60..N160 (more students),
             S3_viol6 (30% violators), S5_boundary (difference at equivalence boundary).
  Analysis variants for hypothesis H1a (explicit defect solution time, Cox model):
     v1     — as defined in analysis_core.py (if robust SE is undefined, p = 1);
     v2     — hybrid: like v1, but when SE is undefined, variance is computed via jackknife over students;
     strat  — Cox model stratified by task (T1/T2) + same fallback jackknife.
  Equivalence testing variants for H1b: two-sample TOST (primary) and paired TOST across student pairs.
  Failure counters: proportion of replications requiring fallback calculation, and failed refit fraction within jackknife.

How to run (from scripts folder; requires sim_core.py, analysis_core.py, this file;
packages: numpy, scipy, pandas, statsmodels, matplotlib):
    python scenarios_simulation.py --selftest                  # code verification (1–2 minutes)
    python scenarios_simulation.py --n-sim 30 --quick          # quick trial of full pipeline
    python scenarios_simulation.py --n-sim 2000 --workers 4    # final run
Results are saved to output/. With the same --seed-base, results do not depend on --workers.
The first 500 replications match the preliminary run (same seeds), so the final run
SUPPLEMENTS preliminary results rather than replacing them with different ones.
Author signature: a1d0s_3l1u45
"""

import argparse
import copy
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from statsmodels.duration.hazard_regression import PHReg

import sim_core as sc
import analysis_core as ac

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TRUTH_PATH = os.path.join(OUT_DIR, "truth_v3.json")
TRUTH_N = 10000            # "very large sample" size for estimating true values
BOUNDARY = ac.EQUIV_BOUND  # 0.25 — equivalence boundary
Z975 = 1.959964
HYP = ["H1a", "H2a_EL", "H2b_GL", "H3"]
PIPES = ("v1", "v2", "strat")


# ----------------------------------------------------------------------------
# 1. Scenarios and parameters
# ----------------------------------------------------------------------------
def build_cells(quick=False):
    """List of "cells": scenario × sample size × number of violators. Numbering sets the seed (do not change!)."""
    cells = [
        dict(name="S0_null", kind="null", n=40, viol=0, pp=False, truth="null"),
        dict(name="S1_planted", kind="planted", n=40, viol=2, pp=True, truth="planted"),
    ]
    for n in ((60, 100) if quick else (60, 80, 100, 120, 160)):
        cells.append(dict(name=f"S2_N{n}", kind="planted", n=n, viol=n // 20, pp=False, truth="planted"))
    cells.append(dict(name="S3_viol6", kind="planted", n=40, viol=6, pp=True, truth=None))
    cells.append(dict(name="S5_boundary", kind="boundary", n=40, viol=0, pp=False, truth="boundary"))
    for i, c in enumerate(cells, start=1):
        c["index"] = i
    if quick:   # under --quick, indices must match full list, otherwise seeds will shift
        full = {c["name"]: c["index"] for c in build_cells(False)}
        for c in cells:
            c["index"] = full[c["name"]]
    return cells


def make_params(kind, n_students, n_viol, boundary_shift=0.0):
    """
    Generator parameters per scenario.
      planted  — planted effects (original PARAMS);
      null     — EG = CG across all parameters, forced "zero solver" disabled;
      boundary — like null, but on hidden tasks EG solves better such that true difference
                 in proportions is at the equivalence boundary (logit shift = boundary_shift).
    """
    P = copy.deepcopy(sc.PARAMS)
    P["n_students"] = n_students
    P["n_protocol_violators"] = n_viol
    if kind in ("null", "boundary"):
        P["p_solvable"]["EG"] = dict(P["p_solvable"]["CG"])
        P["ai_time_reduction"] = {"explicit": 0.0, "hidden": 0.0}
        P["attempts_mean"]["EG"] = dict(P["attempts_mean"]["CG"])
        for name in P["cog_load"]:
            P["cog_load"][name]["mean"]["EG"] = P["cog_load"][name]["mean"]["CG"]
        P["transfer"]["mean"]["EG"] = P["transfer"]["mean"]["CG"]
        P["transfer"]["uptake_penalty"] = 0.0
        P["zero_solver"] = False
    if kind == "boundary":
        for t in ("T3", "T4"):
            P["p_solvable"]["EG"][t] = float(expit(logit(P["p_solvable"]["CG"][t]) + boundary_shift))
    return P


# ----------------------------------------------------------------------------
# 2. Cox model: data, robust SE, fallback jackknife
# ----------------------------------------------------------------------------
def cox_frame(students, tasks):
    """Table for Cox model — identical to analysis_core.test_h1a (explicit tasks, attempted)."""
    s = ac.add_pre_z(students)
    d = tasks[(tasks.task_type == "explicit") & (tasks.attempted == 1)].merge(
        s[["student_id", "pre_z", "eg"]], on="student_id")
    return d.reset_index(drop=True)


def _phreg(t, X, ev, strata):
    """Cox model (Efron ties handling); strata is array of task labels or None."""
    return PHReg(t, X, status=ev, ties="efron", strata=strata)


def cox_robust(d, strata):
    """log HR and cluster-robust SE (cluster = student). SE = NaN if sandwich is undefined."""
    X = d[["eg", "pre_z"]].astype(float).values
    st = d["task"].values if strata else None
    codes = pd.factorize(d["student_id"])[0]
    res = _phreg(d["time_min"].values, X, d["solved"].values, st).fit(groups=codes, disp=False)
    return float(res.params[0]), float(res.bse[0]), res.params


def cox_jackknife(d, strata, start):
    """
    Variance of log HR via delete-one-cluster jackknife:
        Var = (G-1)/G * sum_g (b_(-g) - mean(b_(-g)))^2; p and 95% CI from t-distribution with G-1 df.
    Returns dict; valid=False if too many refits failed (fewer than 80% clusters).
    fail_frac — fraction of failed refits within jackknife (for failure reporting).
    """
    X = d[["eg", "pre_z"]].astype(float).values
    t, ev = d["time_min"].values, d["solved"].values
    st_all = d["task"].values if strata else None
    ids = d["student_id"].values
    uniq = np.unique(ids)
    betas, failed = [], 0
    for g in uniq:
        keep = ids != g
        try:
            r = _phreg(t[keep], X[keep], ev[keep], st_all[keep] if strata else None).fit(start_params=start, disp=False)
            b = float(r.params[0])
            if np.isfinite(b):
                betas.append(b)
            else:
                failed += 1
        except Exception:
            failed += 1
    fail_frac = failed / len(uniq)
    G = len(betas)
    if G < max(5, 0.8 * len(uniq)):
        return dict(valid=False, fail_frac=fail_frac)
    betas = np.array(betas)
    se = float(np.sqrt((G - 1) / G * np.sum((betas - betas.mean()) ** 2)))
    if not np.isfinite(se) or se <= 0:
        return dict(valid=False, fail_frac=fail_frac)
    b_full = float(start[0])
    tcrit = stats.t.ppf(0.975, G - 1)
    return dict(valid=True, fail_frac=fail_frac, p=float(2 * stats.t.sf(abs(b_full / se), G - 1)),
                lo=float(np.exp(b_full - tcrit * se)), hi=float(np.exp(b_full + tcrit * se)))


def h1a_variants(students, tasks, v1, prefix):
    """
    Columns for v2 and strat variants. v1 is result of ac.test_h1a (unchanged).
    Fallback jackknife runs only when sandwich estimate is undefined (faster and saves time).
    """
    d = cox_frame(students, tasks)
    out, fracs, invalid = {}, [], 0

    # --- v2: hybrid ---
    if np.isnan(v1["p"]):
        _, _, params = cox_robust(d, False)
        jk = cox_jackknife(d, False, params)
        fracs.append(jk["fail_frac"])
        if jk["valid"]:
            out.update({f"{prefix}v2_p": jk["p"], f"{prefix}v2_lo": jk["lo"], f"{prefix}v2_hi": jk["hi"]})
        else:
            invalid += 1
            out.update({f"{prefix}v2_p": np.nan, f"{prefix}v2_lo": np.nan, f"{prefix}v2_hi": np.nan})
        out[f"{prefix}v2_fb"] = 1
    else:
        out.update({f"{prefix}v2_p": v1["p"], f"{prefix}v2_lo": v1["hr_ci"][0], f"{prefix}v2_hi": v1["hr_ci"][1],
                    f"{prefix}v2_fb": 0})

    # --- strat: task-stratified ---
    b, se, params = cox_robust(d, True)
    out[f"{prefix}st_hr"] = float(np.exp(b))
    if np.isfinite(se) and se > 0:
        out.update({f"{prefix}st_p": float(2 * stats.norm.sf(abs(b / se))),
                    f"{prefix}st_lo": float(np.exp(b - Z975 * se)), f"{prefix}st_hi": float(np.exp(b + Z975 * se)),
                    f"{prefix}st_fb": 0})
    else:
        jk = cox_jackknife(d, True, params)
        fracs.append(jk["fail_frac"])
        if jk["valid"]:
            out.update({f"{prefix}st_p": jk["p"], f"{prefix}st_lo": jk["lo"], f"{prefix}st_hi": jk["hi"]})
        else:
            invalid += 1
            out.update({f"{prefix}st_p": np.nan, f"{prefix}st_lo": np.nan, f"{prefix}st_hi": np.nan})
        out[f"{prefix}st_fb"] = 1

    out[f"{prefix}jk_fail_frac"] = float(np.mean(fracs)) if fracs else np.nan
    out[f"{prefix}jk_invalid"] = invalid
    return out


# ----------------------------------------------------------------------------
# 3. Paired TOST
# ----------------------------------------------------------------------------
def paired_tost(students, delta=BOUNDARY):
    """
    Paired TOST on differences (EG − CG) in proportion of solved hidden tasks within student pairs.
    Equivalence confirmed if BOTH one-sided tests are significant: p = max(p1, p2) < α.
    """
    w = students.pivot(index="pair_id", columns="group", values="SR_hid")
    diff = (w["EG"] - w["CG"]).dropna().values
    n = len(diff)
    m = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    if n < 3:
        return np.nan, m
    if sd == 0.0:                     # all differences identical: degenerate test
        return (0.0 if abs(m) < delta else 1.0), m
    se = sd / np.sqrt(n)
    p1 = stats.t.sf((m + delta) / se, n - 1)     # H0: difference <= -delta
    p2 = stats.t.cdf((m - delta) / se, n - 1)    # H0: difference >= +delta
    return float(max(p1, p2)), m


# ----------------------------------------------------------------------------
# 4. "True" values from very large sample
# ----------------------------------------------------------------------------
def compute_truth(kind, shift=0.0):
    """
    True (pseudo-true for Cox model) values from dataset of TRUTH_N students.
    Forced "zero solver" is absent in truth: at N = 40 it represents 5% of group and introduces
    a slight bias visible in the bias metric.
    """
    n_viol = TRUTH_N // 20 if kind == "planted" else 0
    P = make_params(kind, TRUTH_N, n_viol, shift)
    P["zero_solver"] = False
    st, tk, _ = sc.generate_dataset(777, params=P)
    mean = lambda col, g: float(st.loc[st.group == g, col].mean())
    d = cox_frame(st, tk)
    X = d[["eg", "pre_z"]].astype(float).values
    b_strat = _phreg(d["time_min"].values, X, d["solved"].values, d["task"].values).fit(disp=False).params[0]
    return {
        "hr": float(ac.test_h1a(st, tk)["hr"]),
        "hr_strat": float(np.exp(b_strat)),
        "tr_b": float(ac.test_h3(st)["coef_eg"]),
        "el_diff": mean("EL", "EG") - mean("EL", "CG"),
        "gl_diff": mean("GL", "EG") - mean("GL", "CG"),
        "hid_diff": mean("SR_hid", "EG") - mean("SR_hid", "CG"),
    }


def calibrate_boundary_shift():
    """Calibrates shift (in logits) at which true difference on hidden tasks equals +0.25."""
    lo, hi, d = 0.0, 3.0, np.nan
    for _ in range(9):
        mid = (lo + hi) / 2.0
        P = make_params("boundary", TRUTH_N, 0, mid)
        st, _, _ = sc.generate_dataset(4242, params=P)
        d = st.loc[st.group == "EG", "SR_hid"].mean() - st.loc[st.group == "CG", "SR_hid"].mean()
        if d < BOUNDARY:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, float(d)


def load_or_compute_truth(force=False):
    """Reads output/truth_v3.json or (1–3 minutes) computes from scratch."""
    if os.path.exists(TRUTH_PATH) and not force:
        with open(TRUTH_PATH, encoding="utf-8") as f:
            return json.load(f)
    print("Computing true values on large sample (once, then cached)...", flush=True)
    shift, d = calibrate_boundary_shift()
    truth = {"boundary_shift": shift, "boundary_diff_achieved": d,
             "null": compute_truth("null"), "planted": compute_truth("planted"),
             "boundary": compute_truth("boundary", shift)}
    with open(TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    return truth


# ----------------------------------------------------------------------------
# 5. Single replication
# ----------------------------------------------------------------------------
def per_protocol(students, tasks):
    """Per-protocol analysis: exclude violators AND their pair partners (paired test remains valid)."""
    bad_pairs = students.loc[students.protocol_violation == 1, "pair_id"].unique()
    st2 = students[~students.pair_id.isin(bad_pairs)]
    tk2 = tasks[tasks.student_id.isin(st2.student_id)]
    return st2, tk2


def analyze(students, tasks, prefix):
    """Full analysis of one dataset: v1 (as is), v2 and strat variants, paired TOST."""
    res = ac.run_confirmatory(students, tasks, light=True)
    h1a, h1b, h2, h3 = res["h1a"], res["h1b"], res["h2"], res["h3"]
    rec = {
        f"{prefix}hr": h1a["hr"], f"{prefix}hr_lo": h1a["hr_ci"][0], f"{prefix}hr_hi": h1a["hr_ci"][1],
        f"{prefix}h1a_p": h1a["p"],
        f"{prefix}tr_b": h3["coef_eg"], f"{prefix}tr_se": h3["se"], f"{prefix}tr_p": h3["p"], f"{prefix}tr_n": h3["n"],
        f"{prefix}el_wp": h2["EL"]["wilcoxon_p"], f"{prefix}el_mp": h2["EL"]["mw_p"], f"{prefix}el_low": h2["EL"]["eg_lower"],
        f"{prefix}gl_wp": h2["GL"]["wilcoxon_p"], f"{prefix}gl_mp": h2["GL"]["mw_p"], f"{prefix}gl_low": h2["GL"]["eg_lower"],
        f"{prefix}il_wp": h2["IL"]["wilcoxon_p"], f"{prefix}il_mp": h2["IL"]["mw_p"],
        f"{prefix}tost_p": h1b["p_tost"], f"{prefix}hid_diff": h1b["diff"],
    }
    pt, pdiff = paired_tost(students)
    rec[f"{prefix}ptost_p"], rec[f"{prefix}pdiff"] = pt, pdiff
    rec.update(h1a_variants(students, tasks, h1a, prefix))
    return rec


def one_rep(job):
    """One replication: generate dataset, analyze (ITT and, if needed, per-protocol)."""
    warnings.filterwarnings("ignore")
    name, kind, n, viol, pp, shift, seed = job
    rec = {"scenario": name, "seed": seed}
    try:
        P = make_params(kind, n, viol, shift)
        students, tasks, _ = sc.generate_dataset(seed, params=P)
        rec.update(analyze(students, tasks, "itt_"))
        if pp:
            st2, tk2 = per_protocol(students, tasks)
            rec.update(analyze(st2, tk2, "pp_"))
        rec["ok"] = 1
    except Exception as e:           # single failed replication should not crash entire run
        rec["ok"] = 0
        rec["error"] = f"{type(e).__name__}: {e}"[:120]
    return rec


# ----------------------------------------------------------------------------
# 6. Metrics
# ----------------------------------------------------------------------------
def holm_matrix(p):
    """Holm adjustment row-by-row (rows = replications, columns = 4 hypotheses). NaN treated as p = 1."""
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    m = p.shape[1]
    order = np.argsort(p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    adj_sorted = np.minimum(np.maximum.accumulate(sorted_p * (m - np.arange(m)), axis=1), 1.0)
    adj = np.empty_like(adj_sorted)
    np.put_along_axis(adj, order, adj_sorted, axis=1)
    return adj


def summarize(df, scenario, n_students, prefix, truth):
    """All metrics for one cell and one analysis type (itt_ or pp_). Long table."""
    d = df[df.ok == 1]
    n_ok = len(d)
    rows = []
    PROB = ("power", "type1", "fwer", "all4", "any_", "equiv", "coverage", "rate", "frac")

    def add(pipeline, variant, metric, value, n=None):
        value = float(value)
        n = n or n_ok
        se = np.sqrt(value * (1 - value) / n) if (0.0 <= value <= 1.0 and metric.startswith(PROB) and n) else np.nan
        rows.append(dict(scenario=scenario, N=n_students, analysis=prefix.strip("_"), pipeline=pipeline,
                         variant=variant, metric=metric, value=value, mc_se=se, n_ok=n))

    col = lambda name: d[f"{prefix}{name}"]
    h1a_cols = {"v1": ("h1a_p", "hr"), "v2": ("v2_p", "hr"), "strat": ("st_p", "st_hr")}

    # --- Tests independent of H1a variant ---
    for variant, ec, gc in (("wilcoxon", "el_wp", "gl_wp"), ("mannwhitney", "el_mp", "gl_mp")):
        for h, c, low in (("H2a_EL", ec, "el_low"), ("H2b_GL", gc, "gl_low")):
            add("-", variant, f"type1_twosided_raw_{h}", (col(c) < ac.ALPHA).mean())
            add("-", variant, f"power_raw_{h}", ((col(c) < ac.ALPHA) & col(low).astype(bool)).mean())
    add("-", "ancova", "type1_twosided_raw_H3", (col("tr_p") < ac.ALPHA).mean())
    add("-", "ancova", "power_raw_H3", ((col("tr_p") < ac.ALPHA) & (col("tr_b") < 0)).mean())
    add("-", "il_null_check", "type1_il_wilcoxon", (col("il_wp") < ac.ALPHA).mean())
    add("-", "il_null_check", "type1_il_mannwhitney", (col("il_mp") < ac.ALPHA).mean())

    # --- Equivalence: two-sample (primary) and paired TOST ---
    add("-", "tost", "equiv_confirmed", (col("tost_p") < ac.ALPHA).mean())
    add("-", "tost", "mean_hid_diff", col("hid_diff").mean())
    pt = col("ptost_p")
    add("-", "tost_paired", "equiv_confirmed", (pt < ac.ALPHA).sum() / max(1, pt.notna().sum()), n=int(pt.notna().sum()))
    add("-", "tost_paired", "mean_hid_diff", col("pdiff").mean())

    # --- H1a variants and Holm family ---
    for pipe, (pc, hc) in h1a_cols.items():
        p_h1a, hr = col(pc), col(hc)
        add(pipe, "cox", "type1_twosided_raw_H1a", (p_h1a < ac.ALPHA).mean())
        add(pipe, "cox", "power_raw_H1a", ((p_h1a < ac.ALPHA) & (hr > 1)).mean())
        for variant, ec, gc in (("wilcoxon", "el_wp", "gl_wp"), ("mannwhitney", "el_mp", "gl_mp")):
            P = np.column_stack([p_h1a, col(ec), col(gc), col("tr_p")]).astype(float)
            D = np.column_stack([hr > 1, col("el_low").astype(bool), col("gl_low").astype(bool), col("tr_b") < 0])
            adj = holm_matrix(P)
            conf = (adj < ac.ALPHA) & D
            for j, h in enumerate(HYP):
                add(pipe, variant, f"power_holm_{h}", conf[:, j].mean())
            add(pipe, variant, "fwer_twosided_holm", (adj < ac.ALPHA).any(axis=1).mean())
            add(pipe, variant, "any_confirmed_holm", conf.any(axis=1).mean())
            add(pipe, variant, "all4_confirmed_holm", conf.all(axis=1).mean())

    # --- Non-convergence and fallback calculations ---
    add("v1", "cox", "nan_rate", col("hr_lo").isna().mean())            # v1: proportion of replications with undefined SE
    add("strat", "cox", "nan_rate", col("st_fb").mean())                # strat: proportion of replications requiring jackknife
    jk = col("jk_fail_frac").dropna()
    add("-", "jackknife", "frac_failed_refits_mean", jk.mean() if len(jk) else np.nan, n=max(1, len(jk)))
    add("-", "jackknife", "frac_reps_with_jackknife", col("jk_fail_frac").notna().mean())
    add("-", "jackknife", "frac_reps_jackknife_invalid", (col("jk_invalid") > 0).mean())

    # --- Coverage and bias (when truth is known) ---
    if truth is not None:
        for pipe, lo_c, hi_c, hr_c, tkey in (("v1", "hr_lo", "hr_hi", "hr", "hr"), ("v2", "v2_lo", "v2_hi", "hr", "hr"),
                                              ("strat", "st_lo", "st_hi", "st_hr", "hr_strat")):
            ok = col(lo_c).notna() & col(hi_c).notna()
            dv = d[ok.values]
            tv = truth[tkey]
            if len(dv):
                cov = ((dv[f"{prefix}{lo_c}"] <= tv) & (tv <= dv[f"{prefix}{hi_c}"])).mean()
                add(pipe, "truth", "coverage95_hr", cov, n=len(dv))
            add(pipe, "truth", "bias_ln_hr", np.log(col(hr_c)).mean() - np.log(tv))
            add(pipe, "truth", f"true_hr", tv)
        tcrit = stats.t.ppf(0.975, (col("tr_n") - 3).clip(lower=1))
        lo, hi = col("tr_b") - tcrit * col("tr_se"), col("tr_b") + tcrit * col("tr_se")
        add("-", "truth", "coverage95_transfer_b", ((lo <= truth["tr_b"]) & (truth["tr_b"] <= hi)).mean())
        add("-", "truth", "bias_transfer_b", col("tr_b").mean() - truth["tr_b"])
        add("-", "truth", "true_transfer_b", truth["tr_b"])
        add("-", "truth", "true_hid_diff", truth["hid_diff"])
    return rows


# ----------------------------------------------------------------------------
# 7. Report and figure
# ----------------------------------------------------------------------------
def cp_ci(rate, n):
    """95% Clopper–Pearson confidence interval for proportion (rate * n successes out of n)."""
    if pd.isna(rate) or np.isnan(rate):
        return np.nan, np.nan
    k = int(round(rate * n))
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95)
    return ci.low, ci.high


def build_report(S, args, truth):
    """Text report with main tables (for inclusion in paper)."""
    def val(scen, metric, pipeline="-", variant=None, an="itt"):
        q = S[(S.scenario == scen) & (S.metric == metric) & (S.pipeline == pipeline) & (S.analysis == an)]
        if variant:
            q = q[q.variant == variant]
        return float(q.value.iloc[0]) if len(q) else np.nan

    reps = S[(S.metric == "equiv_confirmed") & (S.variant == "tost") & (S.analysis == "itt")].set_index("scenario").n_ok
    n_rep = int(reps.max())
    f3 = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    f_ci = lambda lo, hi: f"[{lo:.3f}; {hi:.3f}]" if pd.notna(lo) and pd.notna(hi) else "—"
    L = [f"# Extended simulation results v3 (seed-base = {args.seed_base})\n",
         "Replications per cell: " + ", ".join(f"{k}: {int(v)}" for k, v in reps.items()) + ".\n",
         f"Monte Carlo error for probabilities (95%, worst case p = 0.5): from {1.96 * np.sqrt(0.25 / n_rep):.3f} "
         f"(cells with largest N_rep) to {1.96 * np.sqrt(0.25 / int(reps.min())):.3f}.\n"]
    n0 = int(reps.get("S0_null", n_rep))      # replication count in S0 — for block A confidence intervals

    L.append("## A. Null scenario S0: Type I error rate (two-sided rejections at α = .05) and 95% Clopper–Pearson CI\n")
    rows = []
    for h, pipe, var in (("H1a v1", "v1", "cox"), ("H1a v2", "v2", "cox"), ("H1a strat", "strat", "cox")):
        v = val("S0_null", "type1_twosided_raw_H1a", pipe, var)
        lo, hi = cp_ci(v, n0)
        rows.append({"Test": h, "rate": f3(v), "95% CI": f_ci(lo, hi)})
    for h, var, nm in (("H2a_EL", "wilcoxon", "H2a (Wilcoxon)"), ("H2b_GL", "wilcoxon", "H2b (Wilcoxon)"),
                       ("H2a_EL", "mannwhitney", "H2a (Mann–Whitney)"), ("H2b_GL", "mannwhitney", "H2b (Mann–Whitney)")):
        v = val("S0_null", f"type1_twosided_raw_{h}", "-", var)
        lo, hi = cp_ci(v, n0)
        rows.append({"Test": nm, "rate": f3(v), "95% CI": f_ci(lo, hi)})
    v = val("S0_null", "type1_twosided_raw_H3", "-", "ancova")
    lo, hi = cp_ci(v, n0)
    rows.append({"Test": "H3 (ANCOVA)", "rate": f3(v), "95% CI": f_ci(lo, hi)})
    L.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
    rows = []
    for pipe in PIPES:
        for var, nm in (("wilcoxon", "Wilcoxon"), ("mannwhitney", "Mann–Whitney")):
            v = val("S0_null", "fwer_twosided_holm", pipe, var)
            lo, hi = cp_ci(v, n0)
            rows.append({"H1a variant": pipe, "H2 test": nm, "FWER (Holm)": f3(v), "95% CI": f_ci(lo, hi),
                         "confirmed in hypothesized direction": f3(val("S0_null", "any_confirmed_holm", pipe, var))})
    L.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
    L.append(f"Equivalence under true difference ≈ 0 (TOST power): two-sample {f3(val('S0_null', 'equiv_confirmed', variant='tost'))}, "
             f"paired {f3(val('S0_null', 'equiv_confirmed', variant='tost_paired'))}.\n")

    L.append("## B. Power under planted effects (S1), Holm adjustment, paired Wilcoxon\n")
    rows = []
    for h in HYP:
        r = {"Hypothesis": h}
        for pipe in PIPES:
            r[f"Holm, H1a {pipe}"] = f3(val("S1_planted", f"power_holm_{h}", pipe, "wilcoxon"))
        r["Holm (v2, Mann–Whitney)"] = f3(val("S1_planted", f"power_holm_{h}", "v2", "mannwhitney"))
        rows.append(r)
    r = {"Hypothesis": "all four"}
    for pipe in PIPES:
        r[f"Holm, H1a {pipe}"] = f3(val("S1_planted", "all4_confirmed_holm", pipe, "wilcoxon"))
    r["Holm (v2, Mann–Whitney)"] = f3(val("S1_planted", "all4_confirmed_holm", "v2", "mannwhitney"))
    rows.append(r)
    L.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
    L.append(f"TOST power under true equivalence: two-sample {f3(val('S1_planted', 'equiv_confirmed', variant='tost'))}, "
             f"paired {f3(val('S1_planted', 'equiv_confirmed', variant='tost_paired'))}.\n")

    L.append("## C. Power as a function of sample size (Holm, paired Wilcoxon, H1a v2) and TOST\n")
    rows = []
    for scen in S.scenario.unique():
        if scen.startswith(("S1", "S2")):
            r = {"N": int(S[S.scenario == scen].N.iloc[0])}
            r.update({h: f3(val(scen, f"power_holm_{h}", "v2", "wilcoxon")) for h in HYP})
            r["all four"] = f3(val(scen, "all4_confirmed_holm", "v2", "wilcoxon"))
            r["TOST two-sample"] = f3(val(scen, "equiv_confirmed", variant="tost"))
            r["TOST paired"] = f3(val(scen, "equiv_confirmed", variant="tost_paired"))
            rows.append(r)
    if rows:
        L.append("```\n" + pd.DataFrame(rows).sort_values("N").to_string(index=False) + "\n```\n")

    L.append("## D. ITT and per-protocol (Holm, paired Wilcoxon, H1a v2)\n")
    rows = []
    for scen in ("S1_planted", "S3_viol6"):
        if (S.scenario == scen).any():
            for an in ("itt", "pp"):
                r = {"scenario": scen, "analysis": an}
                r.update({h: f3(val(scen, f"power_holm_{h}", "v2", "wilcoxon", an)) for h in HYP})
                r["TOST two-sample"] = f3(val(scen, "equiv_confirmed", variant="tost", an=an))
                rows.append(r)
    if rows:
        L.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")

    L.append("## E. TOST size at equivalence boundary (S5)\n")
    if (S.scenario == "S5_boundary").any():
        L.append(f"True difference ≈ {truth['boundary']['hid_diff']:.3f} (target 0.25). Proportion of equivalence confirmations: "
                 f"two-sample {f3(val('S5_boundary', 'equiv_confirmed', variant='tost'))}, "
                 f"paired {f3(val('S5_boundary', 'equiv_confirmed', variant='tost_paired'))} (should be ≤ 0.05).\n")

    L.append("## F0. Non-convergence and fallback calculations\n")
    rows = []
    for scen in S.scenario.unique():
        rows.append({"scenario": scen, "N": int(S[S.scenario == scen].N.iloc[0]),
                     "v1: SE undefined": f3(val(scen, "nan_rate", "v1", "cox")),
                     "strat: jackknife required": f3(val(scen, "nan_rate", "strat", "cox")),
                     "jackknife failed refits frac": f3(val(scen, "frac_failed_refits_mean", "-", "jackknife")),
                     "jackknife invalid": f3(val(scen, "frac_reps_jackknife_invalid", "-", "jackknife"))})
    L.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")

    L.append("## F. 95% CI coverage and bias (HR coverage evaluated on replications with defined CI)\n")
    rows = []
    for scen in S.scenario.unique():
        if (S[(S.scenario == scen) & (S.metric == "coverage95_hr")]).empty:
            continue
        rows.append({"scenario": scen, "N": int(S[S.scenario == scen].N.iloc[0]),
                     "HR coverage v1": f3(val(scen, "coverage95_hr", "v1", "truth")),
                     "v2": f3(val(scen, "coverage95_hr", "v2", "truth")),
                     "strat": f3(val(scen, "coverage95_hr", "strat", "truth")),
                     "bias ln HR v2": f3(val(scen, "bias_ln_hr", "v2", "truth")),
                     "bias strat": f3(val(scen, "bias_ln_hr", "strat", "truth")),
                     "coverage b": f3(val(scen, "coverage95_transfer_b", "-", "truth"))})
    L.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")
    L.append("True values: " + "; ".join(
        f"{k}: HR = {truth[k]['hr']:.3f}, HR(strat) = {truth[k]['hr_strat']:.3f}, b = {truth[k]['tr_b']:.3f}, "
        f"hidden diff = {truth[k]['hid_diff']:.3f}" for k in ("null", "planted", "boundary")) + "\n")
    return "\n".join(L)


def make_power_figure(S):
    """Figure: power (Holm, H1a v2) and TOST power as a function of sample size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scen = [s for s in S.scenario.unique() if s.startswith(("S1", "S2"))]
    if len(scen) < 2:
        return

    def get(s, metric, pipe, var):
        r = S[(S.scenario == s) & (S.metric == metric) & (S.pipeline == pipe) & (S.variant == var) & (S.analysis == "itt")]
        return (int(r.N.iloc[0]), float(r.value.iloc[0]), float(r.mc_se.iloc[0])) if len(r) else None

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for h, label in zip(HYP, ["H1a (time)", "H2a (EL)", "H2b (GL)", "H3 (transfer)"]):
        pts = sorted(p for p in (get(s, f"power_holm_{h}", "v2", "wilcoxon") for s in scen) if p)
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[1.96 * p[2] for p in pts], marker="o", label=label, capsize=2)
    for var, style, label in (("tost", "--", "H1b (TOST, two-sample)"), ("tost_paired", ":", "H1b (TOST, paired)")):
        pts = sorted(p for p in (get(s, "equiv_confirmed", "-", var) for s in scen) if p)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", ls=style, color="gray", label=label)
    ax.axhline(0.8, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Total students, N (equal allocation across groups)")
    ax.set_ylabel("Power (with Holm correction)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_power_by_N.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. Self-test
# ----------------------------------------------------------------------------
def selftest():
    """Checks: Holm = statsmodels; null scenario is truly null; jackknife agrees with sandwich;
    stratified Cox works; paired TOST = statsmodels."""
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.weightstats import ttost_paired
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.3, size=(200, 4))
    assert np.allclose(holm_matrix(p), np.array([multipletests(r, method="holm")[1] for r in p])), "Holm mismatch"
    print("OK 1/5: holm_matrix matches statsmodels.multipletests.")

    P = make_params("null", 4000, 0)
    st, tk, _ = sc.generate_dataset(11, params=P)
    r = ac.run_confirmatory(st, tk, light=True)
    mean = lambda col, g: st.loc[st.group == g, col].mean()
    chk = {"|ln HR|": abs(np.log(r["h1a"]["hr"])), "EL EG-CG": mean("EL", "EG") - mean("EL", "CG"),
           "transfer b": r["h3"]["coef_eg"], "hidden EG-CG": r["h1b"]["diff"]}
    print("   null scenario, N = 4000:", {k: round(float(v), 3) for k, v in chk.items()})
    assert chk["|ln HR|"] < 0.12 and abs(chk["EL EG-CG"]) < 0.2 and abs(chk["transfer b"]) < 0.3 and abs(chk["hidden EG-CG"]) < 0.06
    print("OK 2/5: null scenario is truly null.")

    P = make_params("planted", 40, 2)
    ratios = []
    for seed in range(500, 530):
        st, tk, _ = sc.generate_dataset(seed, params=P)
        d = cox_frame(st, tk)
        b, se, params = cox_robust(d, False)
        if not np.isfinite(se):
            continue
        jk = cox_jackknife(d, False, params)
        if jk["valid"]:
            ratios.append((np.log(jk["hi"]) - np.log(jk["lo"])) / (2 * Z975 * se))
    med = float(np.median(ratios))
    print(f"   SE ratio jackknife / sandwich (median over {len(ratios)} replications): {med:.2f}")
    assert 0.8 < med < 1.6, "jackknife substantially diverges from sandwich"
    print("OK 3/5: jackknife agrees with sandwich estimate.")

    b, se, _ = cox_robust(d, True)
    assert np.isfinite(b) and (np.isfinite(se) or True)
    print(f"OK 4/5: stratified Cox model works (log HR = {b:.3f}).")

    x = rng.normal(.55, .4, 20)
    y = x + rng.normal(.03, .3, 20)
    dd = y - x
    n, se_ = len(dd), dd.std(ddof=1) / np.sqrt(len(dd))
    mine = max(stats.t.sf((dd.mean() + .25) / se_, n - 1), stats.t.cdf((dd.mean() - .25) / se_, n - 1))
    assert abs(mine - ttost_paired(y, x, -.25, .25)[0]) < 1e-9
    print("OK 5/5: paired TOST matches statsmodels.ttost_paired.")


# ----------------------------------------------------------------------------
# 9. Main function
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extended simulations v3 (operating characteristics)")
    parser.add_argument("--n-sim", type=int, default=1000, help="replications per scenario (for final: 2000)")
    parser.add_argument("--workers", type=int, default=1, help="number of processes")
    parser.add_argument("--seed-base", type=int, default=500000, help="base seed")
    parser.add_argument("--only", nargs="*", help="run only specified scenarios, e.g. S0_null S1_planted")
    parser.add_argument("--quick", action="store_true", help="reduce N grid (for trial run)")
    parser.add_argument("--recompute-truth", action="store_true", help="recompute true values")
    parser.add_argument("--selftest", action="store_true", help="only verify code and exit")
    parser.add_argument("--skip-existing", action="store_true", help="do not recompute cells with existing output/raw_v3_*.csv")
    parser.add_argument("--from-raw", action="store_true", help="do not simulate, re-assemble report from output/raw_v3_*.csv")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore")   # warnings about tied values on discrete scales are expected
    if args.selftest:
        selftest()
        return

    truth = load_or_compute_truth(force=args.recompute_truth)
    cells = [c for c in build_cells(args.quick) if not args.only or c["name"] in args.only]
    all_rows = []
    for c in cells:
        t0 = time.time()
        shift = truth["boundary_shift"] if c["kind"] == "boundary" else 0.0
        raw_path = os.path.join(OUT_DIR, f"raw_v3_{c['name']}.csv")
        if args.from_raw or (args.skip_existing and os.path.exists(raw_path)):
            df = pd.read_csv(raw_path)
        else:
            jobs = [(c["name"], c["kind"], c["n"], c["viol"], c["pp"], shift,
                     args.seed_base + c["index"] * 1_000_000 + i) for i in range(args.n_sim)]
            recs, batch = [], max(50, args.workers * 25)
            for start in range(0, len(jobs), batch):          # batches to display progress
                part = jobs[start:start + batch]
                if args.workers > 1:
                    with ProcessPoolExecutor(max_workers=args.workers) as ex:
                        recs += list(ex.map(one_rep, part, chunksize=5))
                else:
                    recs += [one_rep(j) for j in part]
                print(f"    {c['name']}: {len(recs)}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
            df = pd.DataFrame(recs)
            df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        n_fail = int((df.ok == 0).sum())
        tr = truth[c["truth"]] if c["truth"] else None
        all_rows += summarize(df, c["name"], c["n"], "itt_", tr)
        if c["pp"]:
            all_rows += summarize(df, c["name"], c["n"], "pp_", None)
        print(f"  {c['name']:12s} N={c['n']:<4d} completed in {time.time() - t0:5.0f} s, failed replications: {n_fail}", flush=True)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(OUT_DIR, "scenarios_summary_v3.csv"), index=False, encoding="utf-8-sig")
    report = build_report(summary, args, truth)
    with open(os.path.join(OUT_DIR, "scenarios_report_v3.md"), "w", encoding="utf-8") as f:
        f.write(report)
    make_power_figure(summary)
    print(report)
    print(f"\nDone. Files in directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
