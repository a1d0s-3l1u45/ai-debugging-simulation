# -*- coding: utf-8 -*-
"""
sim_core.py — SYNTHETIC data generator for the educational example.

IMPORTANT: Data are NOT empirical. Effects are defined in the PARAMS dictionary below
(i.e., "planted" by the simulation author), not discovered in a real
experiment. Use strictly as educational material and ALWAYS with the label
"synthetic data".

What the module does (step by step, per the specification):
  1. Creates 40 students and their baseline test scores (Pre-test, 0–20).
  2. Pairwise randomizes: rank by score -> pairs -> coin toss within pair.
  3. For each student and each of 4 tasks, simulates a 40-minute session
     (time, pytest run count, whether task is solved) under the total session limit.
  4. Simulates 8 survey items (IL, EL, GL; 0–10 scale).
  5. Simulates Transfer test score (0–10).

Calibration check run (optional):  python sim_core.py
"""

import numpy as np
import pandas as pd
from scipy.special import expit, logit  # expit is the sigmoid function, logit is its inverse

_sig = "a1d0s_3l1u45"  # author signature

# ----------------------------------------------------------------------------
# SIMULATION PARAMETERS (approved table; time units are minutes)
# ----------------------------------------------------------------------------
PARAMS = {
    "n_students": 40,          # total students (must be even)
    "pre_mean": 12.0,          # baseline test mean (0–20)
    "pre_sd": 3.5,             # baseline test standard deviation
    "session_minutes": 40.0,   # main round limit
    "p_fixed_order": 0.7,      # proportion of students solving tasks strictly in 1->4 order

    "tasks": ["T1", "T2", "T3", "T4"],
    "task_type": {"T1": "explicit", "T2": "explicit", "T3": "hidden", "T4": "hidden"},

    # Probability that a student is IN PRINCIPLE capable of solving the task
    # (before considering the time limit and before considering the effect of Pre-test).
    "p_solvable": {
        "CG": {"T1": 0.98, "T2": 0.96, "T3": 0.67, "T4": 0.69},
        "EG": {"T1": 0.99, "T2": 0.97, "T3": 0.64, "T4": 0.60},
    },
    "odds_per_sd_pre": 1.4,    # +1 SD on Pre-test -> odds of solving multiplied by 1.4
    "ability_sd": 0.5,         # individual variation in "ability" (logits)

    # Median solution time (for solvers), CG group, minutes.
    "median_time_cg": {"T1": 6.0, "T2": 9.0, "T3": 6.0, "T4": 6.5},
    "time_mult_per_sd_pre": 0.85,   # +1 SD on Pre-test -> time multiplied by 0.85
    "speed_sd": 0.25,          # individual "speed" (log scale)
    "time_sigma": 0.45,        # random time dispersion (log scale)

    # How much AI reduces time in EG (fraction of median under full "uptake")
    "ai_time_reduction": {"explicit": 0.7, "hidden": 0.05},
    "uptake_a": 5.0, "uptake_b": 3.0,  # AI uptake intensity ~ Beta(5, 3)
    "n_protocol_violators": 2,         # prompting protocol violators in EG
    "zero_solver": True,               # True: one CG student solves nothing (as in educational dataset); False — disable (needed for null scenario)

    # If task is unsolvable for student, they "give up" after approximately this many minutes
    "giveup_median": 7.0, "giveup_sigma": 0.3,

    # Average number of pytest runs per task
    "attempts_mean": {
        "CG": {"explicit": 2.6, "hidden": 4.6},
        "EG": {"explicit": 1.8, "hidden": 3.6},
    },
    "attempts_dispersion": 2.0,  # negative binomial dispersion parameter

    # Cognitive load: group means, scale SD (item mean),
    # number of items k, Pre-test effect (points per 1 SD)
    "cog_load": {
        "IL": {"mean": {"CG": 5.8, "EG": 5.8}, "sd": 1.8, "k": 2, "beta_pre": -0.5},
        "EL": {"mean": {"CG": 5.5, "EG": 4.0}, "sd": 1.9, "k": 3, "beta_pre": -0.3},
        "GL": {"mean": {"CG": 6.4, "EG": 5.5}, "sd": 1.8, "k": 3, "beta_pre": 0.3},
    },
    "item_corr": 0.6,  # within-scale item correlation
    "scale_corr": [[1.0, 0.3, 0.1],   # between-scale correlations across IL, EL, GL
                   [0.3, 1.0, -0.3],
                   [0.1, -0.3, 1.0]],

    # Transfer test (0–10)
    "transfer": {
        "mean": {"CG": 6.3, "EG": 4.9}, "sd": 1.9, "r_pre": 0.5,
        "per_hidden_solved": 0.4,   # +0.4 points for each solved hidden task (from 1)
        "uptake_penalty": 1.0,      # greater reliance on AI leads to lower transfer (EG)
    },
}


def generate_dataset(seed=2026, params=None):
    """
    Generates one synthetic dataset.

    Returns three objects:
      students — table "one student = one row" (wide format);
      tasks    — table "one student × one task = one row" (long format);
      meta     — metadata (who is a violator, who is the "zero solver").
    """
    P = params if params is not None else PARAMS
    rng = np.random.default_rng(seed)  # random number generator; seed = reproducibility
    n = P["n_students"]
    if n % 2 != 0:
        raise ValueError("n_students must be even (pairs are formed).")
    tasks = P["tasks"]

    # --- 1. Identifiers and baseline score ------------------------------------
    ids = np.array([f"ID_{k:02d}" for k in range(1, n + 1)])
    pre = np.clip(np.round(rng.normal(P["pre_mean"], P["pre_sd"], n)), 0, 20).astype(int)

    # --- 2. Pairwise randomization (specification section 2) -----------------
    # Sort descending by score; on ties random order.
    order = np.lexsort((rng.random(n), -pre))
    group = np.empty(n, dtype=object)
    pair_id = np.zeros(n, dtype=int)
    for pair_no, k in enumerate(range(0, n, 2), start=1):
        a, b = order[k], order[k + 1]
        if rng.integers(0, 2) == 0:      # "coin toss" within pair
            group[a], group[b] = "CG", "EG"
        else:
            group[a], group[b] = "EG", "CG"
        pair_id[a] = pair_id[b] = pair_no

    z = (pre - P["pre_mean"]) / P["pre_sd"]    # standardized Pre-test

    # --- 3. Individual student characteristics --------------------------------
    speed = rng.normal(0, P["speed_sd"], n)         # fast/slow "by nature"
    ability = rng.normal(0, P["ability_sd"], n)     # "ability" beyond Pre-test
    uptake = rng.beta(P["uptake_a"], P["uptake_b"], n)  # AI uptake (only applied in EG)
    uptake_mean = P["uptake_a"] / (P["uptake_a"] + P["uptake_b"])

    eg_idx = np.where(group == "EG")[0]
    cg_idx = np.where(group == "CG")[0]
    violators = rng.choice(eg_idx, size=P["n_protocol_violators"], replace=False)
    protocol_violation = np.zeros(n, dtype=int)
    protocol_violation[violators] = 1
    uptake[violators] = 0.95                    # violators almost completely rely on AI
    zero_solver = cg_idx[np.argmin(pre[cg_idx])]  # one CG student solves nothing

    # --- 4. Main round: 40 minutes, 4 tasks ----------------------------------
    rows = []
    for i in range(n):
        g = group[i]
        if rng.random() < P["p_fixed_order"]:
            task_order = list(tasks)                   # strictly T1 -> T4
        else:
            task_order = list(rng.permutation(tasks))  # free order
        clock = 0.0  # minutes elapsed in session

        for pos, t in enumerate(task_order, start=1):
            ttype = P["task_type"][t]

            # 4.1. Can the student in principle solve this task?
            if P.get("zero_solver", True) and i == zero_solver:
                solvable = False
            else:
                lp = (logit(P["p_solvable"][g][t])
                      + np.log(P["odds_per_sd_pre"]) * z[i] + ability[i])
                solvable = rng.random() < expit(lp)

            # 4.2. How long would the solution take (if solvable)
            med = P["median_time_cg"][t]
            if g == "EG":
                med *= (1.0 - P["ai_time_reduction"][ttype] * uptake[i])
            solve_time = (med * P["time_mult_per_sd_pre"] ** z[i] * np.exp(speed[i])
                          * np.exp(rng.normal(0, P["time_sigma"])))
            # 4.3. After how many minutes would the student "give up" (if unsolvable)
            giveup = P["giveup_median"] * np.exp(rng.normal(0, P["giveup_sigma"]))

            # 4.4. Consideration of total session time limit
            remaining = P["session_minutes"] - clock
            if remaining <= 0.05:
                # No time remaining: task not reached
                rows.append(dict(student_id=ids[i], task=t, task_type=ttype,
                                 order_position=pos, attempted=0, time_sec=np.nan,
                                 time_min=np.nan, solved=0, attempts_pytest=0,
                                 ended_by="not_reached"))
                continue

            work = solve_time if solvable else giveup
            if work <= remaining:
                t_spent, solved = work, int(solvable)
                ended_by = "solved" if solvable else "abandoned"
                clock += work
            else:
                t_spent, solved, ended_by = remaining, 0, "session_end"
                clock = P["session_minutes"]

            # 4.5. Number of pytest runs (longer work -> more runs)
            mu = P["attempts_mean"][g][ttype]
            typical = P["median_time_cg"][t]
            if g == "EG":
                typical *= (1.0 - P["ai_time_reduction"][ttype] * uptake_mean)
            mu_eff = mu * (max(t_spent, 0.5) / typical) ** 0.4
            if not solved:
                mu_eff *= 1.3
            extra_mean = max(mu_eff - 1.0, 0.2)
            k = P["attempts_dispersion"]
            extra = rng.negative_binomial(k, k / (k + extra_mean))
            attempts = 1 + int(extra)

            sec = int(round(t_spent * 60))
            rows.append(dict(student_id=ids[i], task=t, task_type=ttype,
                             order_position=pos, attempted=1, time_sec=sec,
                             time_min=round(sec / 60.0, 3), solved=solved,
                             attempts_pytest=attempts, ended_by=ended_by))

    tasks_df = pd.DataFrame(rows)
    info = pd.DataFrame({"student_id": ids, "group": group, "pair_id": pair_id})
    tasks_df = tasks_df.merge(info, on="student_id")
    tasks_df = tasks_df[["student_id", "pair_id", "group", "task", "task_type",
                         "order_position", "attempted", "time_sec", "time_min",
                         "solved", "attempts_pytest", "ended_by"]]
    tasks_df["synthetic"] = True

    # Student summary from long table
    solved_by_type = tasks_df.pivot_table(index="student_id", columns="task_type",
                                          values="solved", aggfunc="sum")
    sr_expl_n = solved_by_type.loc[ids, "explicit"].values
    sr_hid_n = solved_by_type.loc[ids, "hidden"].values
    attempts_total = tasks_df.groupby("student_id")["attempts_pytest"].sum().loc[ids].values

    # --- 5. Leppink survey (adaptation, 8 items, 0–10 scale) ------------------
    r = P["item_corr"]
    corr = np.array(P["scale_corr"])
    eps = rng.multivariate_normal(np.zeros(3), corr, size=n)  # correlated scale "tails"
    items = {}
    for j, (name, c) in enumerate(P["cog_load"].items()):  # order: IL, EL, GL
        k_items, sd_t = c["k"], c["sd"]
        # item variance so that scale mean SD equals sd_t
        V = sd_t ** 2 / (r + (1 - r) / k_items)
        latent_total_sd = np.sqrt(r * V)
        resid_sd = np.sqrt(max(latent_total_sd ** 2 - c["beta_pre"] ** 2, 1e-6))
        item_noise_sd = np.sqrt((1 - r) * V)
        mu = np.array([c["mean"][g] for g in group])
        latent = mu + c["beta_pre"] * z + resid_sd * eps[:, j]
        for item_no in range(1, k_items + 1):
            raw = latent + rng.normal(0, item_noise_sd, n)
            items[f"{name}{item_no}"] = np.clip(np.round(raw), 0, 10).astype(int)

    # --- 6. Transfer test (0–10) ---------------------------------------------
    tr = P["transfer"]
    mu_tr = np.array([tr["mean"][g] for g in group])
    noise = rng.normal(0, 1, n)
    is_eg = (group == "EG").astype(float)
    score = (mu_tr
             + tr["sd"] * (tr["r_pre"] * z + np.sqrt(1 - tr["r_pre"] ** 2) * noise)
             + tr["per_hidden_solved"] * (sr_hid_n - 1.0)
             - tr["uptake_penalty"] * (uptake - uptake_mean) * is_eg)
    transfer = np.clip(np.round(score), 0, 10).astype(int)

    # --- 7. Assemble student table -------------------------------------------
    students = pd.DataFrame({
        "student_id": ids, "pair_id": pair_id, "group": group, "pre_score": pre,
        "protocol_violation": protocol_violation,
        "ai_prompts_n": np.where(group == "EG",
                                  rng.poisson(3 + 12 * uptake), np.nan),
        "SR_expl_n": sr_expl_n, "SR_hid_n": sr_hid_n,
        "SR_expl": sr_expl_n / 2.0, "SR_hid": sr_hid_n / 2.0,
        "attempts_total": attempts_total,
    })
    for key, vals in items.items():
        students[key] = vals
    students["IL"] = students[["IL1", "IL2"]].mean(axis=1)
    students["EL"] = students[["EL1", "EL2", "EL3"]].mean(axis=1)
    students["GL"] = students[["GL1", "GL2", "GL3"]].mean(axis=1)
    students["transfer_score"] = transfer
    students["synthetic"] = True

    meta = {"seed": seed,
            "protocol_violators": list(ids[violators]),
            "zero_solver_id": str(ids[zero_solver])}
    return students, tasks_df, meta


# Data dictionary for export (Methods section / Appendix)
DATA_DICTIONARY = [
    ("student_id", "De-identified ID (ID_01...ID_40)", "string"),
    ("pair_id", "Matched pair number (by Pre-test)", "1–20"),
    ("group", "Group: CG — control, EG — experimental", "CG/EG"),
    ("pre_score", "Baseline test (Pre-test)", "0–20, integer"),
    ("protocol_violation", "Prompting protocol violation (from logs)", "0/1"),
    ("ai_prompts_n", "Number of AI prompts during session (EG only)", "integer"),
    ("SR_expl_n / SR_hid_n", "Number of solved explicit / hidden tasks", "0–2"),
    ("SR_expl / SR_hid", "Proportion of solved explicit / hidden tasks", "0, 0.5, 1"),
    ("attempts_total", "Total pytest runs during session", "integer"),
    ("IL1–IL2, EL1–EL3, GL1–GL3", "Adapted Leppink survey items", "0–10, integer"),
    ("IL / EL / GL", "Subscale item average", "0–10"),
    ("transfer_score", "Transfer test score (without AI)", "0–10, integer"),
    ("task", "T1, T2 — explicit defects; T3, T4 — hidden semantic defects", "T1–T4"),
    ("task_type", "explicit — explicit defect; hidden — hidden semantic defect", "string"),
    ("order_position", "Order position in which student started the task", "1–4"),
    ("attempted", "Whether student started the task (0 — ran out of time)", "0/1"),
    ("time_sec / time_min", "Time from opening file to success (or abandonment/session end)", "sec / min"),
    ("solved", "Task solved (pytest 100%); 0 = censored", "0/1"),
    ("ended_by", "solved / abandoned / session_end / not_reached", "string"),
    ("synthetic", "LABEL: synthetic data, not empirical", "True"),
]


def calibrate(n_rep=300, seed0=1000):
    """
    Check: how close "realized" (actual) values match the approved table.
    Needed because the 40-minute limit affects the final proportion of solved tasks.
    """
    rec = []
    for s in range(n_rep):
        st, tk, _ = generate_dataset(seed0 + s)
        d = tk[tk.attempted == 1]
        row = {}
        for g in ("CG", "EG"):
            for t in PARAMS["tasks"]:
                sub = tk[(tk.group == g) & (tk.task == t)]
                row[f"solved_{g}_{t}"] = sub["solved"].mean()
                subs = d[(d.group == g) & (d.task == t) & (d.solved == 1)]
                row[f"medtime_{g}_{t}"] = subs["time_min"].median()
            row[f"attempts_{g}"] = tk[(tk.group == g) & (tk.attempted == 1)]["attempts_pytest"].mean()
            for sc in ("IL", "EL", "GL", "transfer_score"):
                row[f"{sc}_{g}"] = st.loc[st.group == g, sc].mean()
        rec.append(row)
    return pd.DataFrame(rec).mean()


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("Realized means over", 300, "replications (calibration check):")
    print(calibrate().round(2).to_string())
