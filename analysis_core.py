# -*- coding: utf-8 -*-
"""
analysis_core.py — statistical analysis following the revised specification (sections 3–6).

Hypothesis testing functions collected here. Used by two scripts:
  run_demo.py          — single dataset + report + figures;
  power_simulation.py  — multiple replications for power estimation.

Hypotheses (all tests two-sided, α = 0.05, Holm correction for H1a, H2a, H2b, H3):
  H1a — EG solves explicit defects faster (T1–T2)           -> Cox regression
  H1b — groups are equivalent on hidden defects (T3–T4)     -> TOST, bounds ±0.25
  H2a — EL in EG is lower, H2b — GL in EG is lower          -> Wilcoxon signed-rank test on pairs
  H3  — Transfer is higher in CG                            -> ANCOVA (Pre-test covariate)
Author signature: a1d0s_3l1u45
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff
from statsmodels.stats.weightstats import ttost_ind
from statsmodels.stats.multitest import multipletests

EQUIV_BOUND = 0.25   # equivalence bounds for H1b (proportion of solved hidden tasks)
ALPHA = 0.05


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def add_pre_z(students):
    """Adds standardized Pre-test (pre_z) — needed as a covariate."""
    s = students.copy()
    s["pre_z"] = (s["pre_score"] - s["pre_score"].mean()) / s["pre_score"].std(ddof=1)
    s["eg"] = (s["group"] == "EG").astype(int)
    return s


def z_from_p(p):
    """Approximate z from two-sided p-value (needed for effect size r = Z / sqrt(N))."""
    p = min(max(p, 1e-300), 1.0)
    return float(stats.norm.isf(p / 2.0))


def cronbach_alpha(items_df):
    """Cronbach's α for a set of items (columns = items)."""
    k = items_df.shape[1]
    item_var = items_df.var(axis=0, ddof=1).sum()
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return float(k / (k - 1) * (1 - item_var / total_var))


def ancova(df, y_col):
    """
    ANCOVA: y ~ group + Pre-test. Returns group effect estimate (EG − CG),
    p-value, partial η², adjusted means, and assumption checks.
    """
    d = df[[y_col, "eg", "pre_score"]].dropna().copy()
    model = smf.ols(f"{y_col} ~ eg + pre_score", data=d).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_g, ss_r = aov.loc["eg", "sum_sq"], aov.loc["Residual", "sum_sq"]
    mean_pre = d["pre_score"].mean()
    adj_cg = float(model.predict(pd.DataFrame({"eg": [0], "pre_score": [mean_pre]}))[0])
    adj_eg = float(model.predict(pd.DataFrame({"eg": [1], "pre_score": [mean_pre]}))[0])
    return {
        "n": int(len(d)),
        "coef_eg": float(model.params["eg"]),
        "se": float(model.bse["eg"]),
        "t": float(model.tvalues["eg"]),
        "p": float(model.pvalues["eg"]),
        "eta2_p": float(ss_g / (ss_g + ss_r)),
        "adj_mean_cg": adj_cg, "adj_mean_eg": adj_eg,
        "shapiro_resid_p": float(stats.shapiro(model.resid).pvalue),
        "levene_p": float(stats.levene(d.loc[d.eg == 1, y_col], d.loc[d.eg == 0, y_col]).pvalue),
    }


# ----------------------------------------------------------------------------
# H1a: solution time for explicit defects (survival analysis with censoring)
# ----------------------------------------------------------------------------
def test_h1a(students, tasks):
    s = add_pre_z(students)
    d = tasks[(tasks.task_type == "explicit") & (tasks.attempted == 1)].merge(
        s[["student_id", "pre_z", "eg"]], on="student_id")
    d = d.reset_index(drop=True)

    # Cox regression: event = "task solved"; censoring = abandoned / session end.
    # Standard errors are robust to clustering by student (2 tasks per student).
    X = d[["eg", "pre_z"]].astype(float).values
    codes = pd.factorize(d["student_id"])[0]
    mod = PHReg(d["time_min"].values, X, status=d["solved"].values, ties="efron")
    res = mod.fit(groups=codes)
    ci = res.conf_int()
    out = {
        "hr": float(np.exp(res.params[0])),
        "hr_ci": (float(np.exp(ci[0, 0])), float(np.exp(ci[0, 1]))),
        "p": float(res.pvalues[0]),
        "n_obs": int(len(d)),
        "n_events": int(d["solved"].sum()),
    }

    # Descriptives for each task: Kaplan–Meier median and log-rank test
    per_task = []
    for t in ("T1", "T2"):
        dt = d[d.task == t]
        row = {"task": t}
        for g in ("CG", "EG"):
            dg = dt[dt.group == g]
            sf = SurvfuncRight(dg["time_min"].values, dg["solved"].values)
            row[f"km_median_{g}"] = float(sf.quantile(0.5)) if len(dg) else np.nan
            row[f"solved_{g}"] = int(dg["solved"].sum())
            row[f"n_{g}"] = int(len(dg))
        try:
            chi2, p_lr = survdiff(dt["time_min"].values, dt["solved"].values,
                                  dt["eg"].values)
            row["logrank_p"] = float(p_lr)
        except Exception:
            row["logrank_p"] = np.nan
        per_task.append(row)
    out["per_task"] = pd.DataFrame(per_task)

    # Secondary analysis: ANCOVA on mean ln(time) of SOLVED explicit tasks
    ds = d[d.solved == 1].copy()
    ds["ln_time"] = np.log(ds["time_min"])
    agg = ds.groupby("student_id")["ln_time"].mean().rename("ln_T_expl").reset_index()
    sec = s.merge(agg, on="student_id", how="inner")
    out["secondary_ancova"] = ancova(sec, "ln_T_expl")
    return out


# ----------------------------------------------------------------------------
# H1b: equivalence on hidden defects (TOST)
# ----------------------------------------------------------------------------
def test_h1b(students, tasks, light=False):
    s = add_pre_z(students)
    eg = s.loc[s.group == "EG", "SR_hid"].values
    cg = s.loc[s.group == "CG", "SR_hid"].values
    diff = eg.mean() - cg.mean()

    # 90% confidence interval of difference (Welch) — equivalence if within ±bounds
    v1, v2 = eg.var(ddof=1) / len(eg), cg.var(ddof=1) / len(cg)
    se = np.sqrt(v1 + v2)
    dfw = (v1 + v2) ** 2 / (v1 ** 2 / (len(eg) - 1) + v2 ** 2 / (len(cg) - 1))
    tcrit = stats.t.ppf(0.95, dfw)
    ci90 = (float(diff - tcrit * se), float(diff + tcrit * se))

    try:
        p_tost = float(ttost_ind(eg, cg, -EQUIV_BOUND, EQUIV_BOUND, usevar="unequal")[0])
    except Exception:
        p_tost = np.nan
    out = {"diff": float(diff), "ci90": ci90, "p_tost": p_tost,
           "equivalent": bool(p_tost < ALPHA), "bound": EQUIV_BOUND,
           "mean_cg": float(cg.mean()), "mean_eg": float(eg.mean())}

    if not light:
        # Supplementary: GEE logistic model on individual tasks (T3, T4)
        hid = tasks[tasks.task_type == "hidden"].merge(
            s[["student_id", "pre_z", "eg"]], on="student_id").reset_index(drop=True)
        hid["sid"] = pd.factorize(hid["student_id"])[0]
        try:
            gee = smf.gee("solved ~ eg + pre_z", groups="sid", data=hid,
                          family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            out["gee_or"] = float(np.exp(gee.params["eg"]))
            out["gee_p"] = float(gee.pvalues["eg"])
        except Exception:
            out["gee_or"], out["gee_p"] = np.nan, np.nan
        # Permutation test on number of solved hidden tasks (0, 1, 2)
        try:
            pt = stats.permutation_test(
                (s.loc[s.group == "EG", "SR_hid_n"].values,
                 s.loc[s.group == "CG", "SR_hid_n"].values),
                lambda x, y: x.mean() - y.mean(),
                permutation_type="independent", n_resamples=9999,
                alternative="two-sided", random_state=1)
            out["perm_p"] = float(pt.pvalue)
        except Exception:
            out["perm_p"] = np.nan
    return out


# ----------------------------------------------------------------------------
# H2: cognitive load subscales (paired Wilcoxon + Mann–Whitney)
# ----------------------------------------------------------------------------
def test_h2(students):
    n_obs = len(students)  # N in formula r = Z / sqrt(N): number of observations (40)
    out = {}
    for sc in ("IL", "EL", "GL"):
        wide = students.pivot(index="pair_id", columns="group", values=sc)
        diff = (wide["EG"] - wide["CG"]).values
        res = {"mean_diff_pairs": float(np.mean(diff))}
        try:
            w = stats.wilcoxon(diff)               # zero differences are discarded
            res["wilcoxon_p"] = float(w.pvalue)
        except ValueError:                          # all differences are zero
            res["wilcoxon_p"] = 1.0
        res["r_wilcoxon"] = z_from_p(res["wilcoxon_p"]) / np.sqrt(n_obs)
        eg = students.loc[students.group == "EG", sc].values
        cg = students.loc[students.group == "CG", sc].values
        mw = stats.mannwhitneyu(eg, cg, alternative="two-sided")
        res["mw_u"] = float(mw.statistic)
        res["mw_p"] = float(mw.pvalue)
        res["r_mw"] = z_from_p(res["mw_p"]) / np.sqrt(n_obs)
        res["eg_lower"] = bool(np.mean(diff) < 0)
        for g, arr in (("CG", cg), ("EG", eg)):
            res[f"median_{g}"] = float(np.median(arr))
            res[f"q1_{g}"] = float(np.percentile(arr, 25))
            res[f"q3_{g}"] = float(np.percentile(arr, 75))
        out[sc] = res
    return out


# ----------------------------------------------------------------------------
# H3: Transfer test (ANCOVA)
# ----------------------------------------------------------------------------
def test_h3(students):
    return ancova(add_pre_z(students), "transfer_score")


# ----------------------------------------------------------------------------
# Full pipeline + Holm correction
# ----------------------------------------------------------------------------
def run_confirmatory(students, tasks, light=False):
    h1a = test_h1a(students, tasks)
    h1b = test_h1b(students, tasks, light=light)
    h2 = test_h2(students)
    h3 = test_h3(students)

    names = ["H1a", "H2a (EL)", "H2b (GL)", "H3"]
    p_raw = np.array([h1a["p"], h2["EL"]["wilcoxon_p"], h2["GL"]["wilcoxon_p"], h3["p"]])
    p_raw = np.nan_to_num(p_raw, nan=1.0)
    rejected, p_holm, _, _ = multipletests(p_raw, alpha=ALPHA, method="holm")
    direction_ok = np.array([h1a["hr"] > 1, h2["EL"]["eg_lower"],
                             h2["GL"]["eg_lower"], h3["coef_eg"] < 0])
    table = pd.DataFrame({
        "hypothesis": names, "p_raw": p_raw, "p_holm": p_holm,
        "direction_as_hypothesized": direction_ok,
        "confirmed": rejected & direction_ok,
    })
    return {"h1a": h1a, "h1b": h1b, "h2": h2, "h3": h3, "holm_table": table}


def pretest_balance(students):
    """Checks Pre-test balance across groups: paired t-test and means."""
    wide = students.pivot(index="pair_id", columns="group", values="pre_score")
    t = stats.ttest_rel(wide["EG"], wide["CG"])
    return {"mean_cg": float(wide["CG"].mean()), "mean_eg": float(wide["EG"].mean()),
            "paired_t_p": float(t.pvalue)}
