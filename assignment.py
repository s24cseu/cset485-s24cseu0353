# cset485-s24cseu0353

"""
CSET485 AI and Society - Assignment #1
German Credit Model Stress-Testing & Fairness Notes
Roll No: 0353 | Seed: random_state = 353 | Split: 70/30
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

RANDOM_STATE = 353

COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment_length", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "target_raw"
]

NOMINAL_COLS = [
    "checking_account", "credit_history", "purpose", "personal_status_sex",
    "other_debtors", "property", "other_installment_plans", "housing",
    "job", "telephone", "foreign_worker"
]

NUMERIC_COLS = [
    "duration", "credit_amount", "age", "installment_rate",
    "residence_since", "existing_credits", "num_dependents"
]

# Ordinal mappings (tiers 0 to 4)
SAVINGS_MAP = {
    "A65": 0,  # unknown / no savings account
    "A61": 1,  # < 100 DM
    "A62": 2,  # 100 <= ... < 500 DM
    "A63": 3,  # 500 <= ... < 1000 DM
    "A64": 4,  # >= 1000 DM
}

EMPLOYMENT_MAP = {
    "A71": 0,  # unemployed
    "A72": 1,  # < 1 year
    "A73": 2,  # 1 <= ... < 4 years
    "A74": 3,  # 4 <= ... < 7 years
    "A75": 4,  # >= 7 years
}


def load_data(local_path: str = "german.data") -> pd.DataFrame:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    try:
        df = pd.read_csv(url, sep=r"\s+", header=None, names=COLUMNS)
    except Exception:
        df = pd.read_csv(local_path, sep=r"\s+", header=None, names=COLUMNS)

    # Map raw target: 1 (good) -> 1 (approve), 2 (bad) -> 0 (reject)
    df["target"] = df["target_raw"].map({1: 1, 2: 0})
    df.drop(columns=["target_raw"], inplace=True)

    # Preserve A14 and A65 as explicit categories (no blank NaNs in raw data)
    for col in NOMINAL_COLS + ["savings", "employment_length"]:
        df[col] = df[col].astype(str).str.strip()

    df["savings_ord"] = df["savings"].map(SAVINGS_MAP)
    df["employment_ord"] = df["employment_length"].map(EMPLOYMENT_MAP)
    return df


def preprocess_split(df: pd.DataFrame):
    raw_age = df["age"].copy()
    raw_credit_amount = df["credit_amount"].copy()
    raw_checking = df["checking_account"].copy()
    raw_sex_status = df["personal_status_sex"].copy()

    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE, stratify=df["target"]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Cap outliers at 99th percentile of training set
    for col in ["credit_amount", "duration"]:
        cap_val = np.percentile(train_df[col], 99)
        print(f"99th percentile cap for {col} (train): {cap_val:.2f}")
        train_df[col] = np.clip(train_df[col], None, cap_val)
        test_df[col] = np.clip(test_df[col], None, cap_val)

    ordinal_cols = ["savings_ord", "employment_ord"]
    scale_cols = NUMERIC_COLS + ordinal_cols

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[scale_cols]),
        columns=scale_cols,
        index=train_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[scale_cols]),
        columns=scale_cols,
        index=test_df.index,
    )

    # One-hot encode nominal columns with drop_first=True
    full_dummies = pd.get_dummies(df[NOMINAL_COLS], columns=NOMINAL_COLS, drop_first=True, dtype=float)
    X_train_cat = full_dummies.loc[train_df.index]
    X_test_cat = full_dummies.loc[test_df.index]

    X_train = pd.concat([train_scaled, X_train_cat], axis=1)
    X_test = pd.concat([test_scaled, X_test_cat], axis=1)
    y_train = train_df["target"]
    y_test = test_df["target"]

    meta = {
        "raw_age": raw_age,
        "raw_credit_amount": raw_credit_amount,
        "raw_checking": raw_checking,
        "raw_sex_status": raw_sex_status,
        "train_df": train_df,
        "test_df": test_df,
        "scaler": scaler,
        "scale_cols": scale_cols,
    }
    return X_train, X_test, y_train, y_test, meta


def part1_models_and_thresholds(X_train, X_test, y_train, y_test):
    print("\n=== PART 1: BUILD & THRESHOLD COMPARISON ===")
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_ols_pred = ols.predict(X_test)

    ols_median = np.median(y_ols_pred)
    print(f"OLS range: [{y_ols_pred.min():.2f}, {y_ols_pred.max():.2f}], median: {ols_median:.3f}")
    ols_flag = (y_ols_pred >= ols_median).astype(int)

    logit = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    logit.fit(X_train, y_train)
    p_logit = logit.predict_proba(X_test)[:, 1]
    logit_flag = (p_logit >= 0.50).astype(int)

    ct = pd.crosstab(
        pd.Series(ols_flag, index=X_test.index, name=f"OLS >= {ols_median:.3f}"),
        pd.Series(logit_flag, index=X_test.index, name="Logit >= 0.50"),
        margins=True,
    )
    print("\nTest-Set Comparison (n=300):")
    print(ct)

    mismatches = int((ols_flag != logit_flag).sum())
    agreements = int((ols_flag == logit_flag).sum())
    print(f"Agreements: {agreements}/300 ({agreements/3:.2f}%), Mismatches: {mismatches}/300 ({mismatches/3:.2f}%)")
    return ols, logit, y_ols_pred, p_logit


def part2_1_age_subgroup_analysis(df: pd.DataFrame):
    print("\n=== PART 2.1: SIMPSON'S PARADOX / AGE SUBGROUP ANALYSIS ===")
    sub_data = df.copy()
    sub_data["duration_std"] = (sub_data["duration"] - sub_data["duration"].mean()) / sub_data["duration"].std()

    groups = {
        "1. Full Dataset": sub_data,
        "2. Subgroup A (Age < 30)": sub_data[sub_data["age"] < 30],
        "3. Subgroup B (Age >= 30)": sub_data[sub_data["age"] >= 30],
    }

    for name, gdf in groups.items():
        X = sm.add_constant(gdf[["duration_std"]])
        model = sm.Logit(gdf["target"], X).fit(disp=False)
        coef = model.params["duration_std"]
        ci_low, ci_high = model.conf_int().loc["duration_std"]
        pval = model.pvalues["duration_std"]
        app_rate = gdf["target"].mean() * 100
        print(
            f"{name:26s} | n={len(gdf):4d} | app={app_rate:5.1f}% | "
            f"beta={coef:+.3f} | 95% CI=[{ci_low:+.3f}, {ci_high:+.3f}] | p={pval:.4f}"
        )


def part2_2_omitted_variable_bias(df: pd.DataFrame):
    print("\n=== PART 2.2: OMITTED-VARIABLE BIAS (DROPPING checking_account) ===")
    # Fit standardized logistic regression on full dataset (1,000 rows) with and without checking_account
    work = df.copy()
    work["savings_ge_1000"] = (work["savings"] == "A64").astype(float)

    features_num = ["age", "credit_amount", "duration", "installment_rate", "savings_ge_1000"]
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(work[features_num]), columns=features_num, index=work.index)

    chk_dummies = pd.get_dummies(work["checking_account"], prefix="chk", drop_first=True, dtype=float)
    other_nom = pd.get_dummies(
        work[[c for c in NOMINAL_COLS if c != "checking_account"]], drop_first=True, dtype=float
    )
    other_num = pd.DataFrame(
        scaler.fit_transform(work[["residence_since", "existing_credits", "num_dependents", "employment_ord"]]),
        columns=["residence_since", "existing_credits", "num_dependents", "employment_ord"],
        index=work.index,
    )

    X_full = pd.concat([X_std, other_num, chk_dummies, other_nom], axis=1)
    X_reduced = pd.concat([X_std, other_num, other_nom], axis=1)

    m_full = LogisticRegression(max_iter=2000, penalty=None, solver="lbfgs").fit(X_full, work["target"])
    m_red = LogisticRegression(max_iter=2000, penalty=None, solver="lbfgs").fit(X_reduced, work["target"])

    s_full = pd.Series(m_full.coef_[0], index=X_full.columns)
    s_red = pd.Series(m_red.coef_[0], index=X_reduced.columns)

    for feat in features_num:
        b_f = s_full[feat]
        b_r = s_red[feat]
        delta = b_r - b_f
        pct_change = ((b_r - b_f) / abs(b_f)) * 100.0
        print(
            f"{feat:18s} | Full={b_f:+.3f} | Reduced={b_r:+.3f} | "
            f"Delta={delta:+.3f} | %Change={pct_change:+.2f}%"
        )


def part2_3_adversarial_subset(p_logit: np.ndarray, y_test: pd.Series, meta: dict):
    print("\n=== PART 2.3: ADVERSARIAL SUBSET CONSTRUCTION ===")
    p_series = pd.Series(p_logit, index=y_test.index)
    conf_fp = (p_series >= 0.75) & (y_test == 0)
    conf_fn = (p_series <= 0.25) & (y_test == 1)
    conf_err_idx = y_test.index[conf_fp | conf_fn]

    print(
        f"Confident errors: {len(conf_err_idx)}/300 ({len(conf_err_idx)/3:.2f}%) | "
        f"Confident FP: {conf_fp.sum()} | Confident FN: {conf_fn.sum()}"
    )

    sample_n = min(50, len(conf_err_idx))
    sample_idx = pd.Series(conf_err_idx).sample(n=sample_n, random_state=RANDOM_STATE)

    ages = meta["raw_age"].loc[sample_idx]
    test_ages = meta["raw_age"].loc[y_test.index]
    sex_status = meta["raw_sex_status"].loc[sample_idx]
    test_sex_status = meta["raw_sex_status"].loc[y_test.index]
    loans = meta["raw_credit_amount"].loc[sample_idx]
    chk = meta["raw_checking"].loc[sample_idx]

    # Male codes in UCI German credit: A91, A93, A94; Female codes: A92, A95
    male_codes = {"A91", "A93", "A94"}
    sample_male_pct = sex_status.isin(male_codes).mean() * 100
    test_male_pct = test_sex_status.isin(male_codes).mean() * 100

    pattern_mask = (loans >= 3000) | (chk.isin(["A12", "A14"]))
    print(f"Sample (n={sample_n}) mean age: {ages.mean():.1f} (test: {test_ages.mean():.1f})")
    print(f"Sample <30: {(ages < 30).mean()*100:.1f}% (test: {(test_ages < 30).mean()*100:.1f}%)")
    print(
        f"Sample Male/Female: {sample_male_pct:.1f}% / {100 - sample_male_pct:.1f}% "
        f"(test: {test_male_pct:.1f}% / {100 - test_male_pct:.1f}%)"
    )
    print(f"Large loan (>=3000 DM) or A12/A14 checking: {pattern_mask.sum()}/{sample_n} ({pattern_mask.mean()*100:.1f}%)")


def fairness_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, is_young: np.ndarray, t: float):
    y_pred = (y_prob >= t).astype(int)

    # Demographic Parity ratio: min(P(y_hat=1|g)) / max(P(y_hat=1|g))
    dp_rates = []
    tpr_list, fpr_list, ppv_list = [], [], []

    for g_mask in [is_young, ~is_young]:
        y_t_g = y_true[g_mask]
        y_p_g = y_pred[g_mask]

        app_rate = y_p_g.mean() if len(y_p_g) > 0 else 0.0
        dp_rates.append(app_rate)

        pos_mask = y_t_g == 1
        neg_mask = y_t_g == 0
        pred_pos_mask = y_p_g == 1

        tpr = y_p_g[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
        fpr = y_p_g[neg_mask].mean() if neg_mask.sum() > 0 else 0.0
        ppv = y_t_g[pred_pos_mask].mean() if pred_pos_mask.sum() > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)
        ppv_list.append(ppv)

    dp_ratio = (min(dp_rates) / max(dp_rates)) if max(dp_rates) > 0 else 0.0
    tpr_ratio = (min(tpr_list) / max(tpr_list)) if max(tpr_list) > 0 else 0.0
    fpr_ratio = (min(fpr_list) / max(fpr_list)) if max(fpr_list) > 0 else 0.0
    eo_ratio = 0.5 * (tpr_ratio + fpr_ratio)
    pp_ratio = (min(ppv_list) / max(ppv_list)) if max(ppv_list) > 0 else 0.0

    return round(dp_ratio, 2), round(eo_ratio, 2), round(pp_ratio, 2)


def part2_4_fairness_tradeoff(p_logit: np.ndarray, y_test: pd.Series, meta: dict):
    print("\n=== PART 2.4: FAIRNESS TRADE-OFF ANALYSIS (t = 0.05 .. 0.95) ===")
    is_young = (meta["raw_age"].loc[y_test.index] < 30).to_numpy()
    y_true = y_test.to_numpy()

    thresholds = np.arange(0.05, 1.00, 0.10)
    rows = []
    for t in thresholds:
        dp, eo, pp = fairness_metrics_at_threshold(y_true, p_logit, is_young, float(t))
        rows.append({"t": round(float(t), 2), "DP": dp, "EO": eo, "PP": pp})

    fair_df = pd.DataFrame(rows)
    print(fair_df.to_string(index=False))

    # Plot DP vs EO across thresholds
    plt.figure(figsize=(8, 4.5))
    plt.plot(fair_df["DP"], fair_df["EO"], marker="o", linewidth=2)
    for _, r in fair_df.iterrows():
        label = f"t={r['t']:.2f}" if r["t"] in (0.05, 0.35, 0.95) else f"{r['t']:.2f}"
        plt.annotate(label, (r["DP"], r["EO"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xlabel("Demographic Parity (DP)")
    plt.ylabel("Equalized Odds (EO)")
    plt.title("DP vs. EO Across Thresholds (Age < 30 vs. Age >= 30)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dp_vs_eo_tradeoff.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    X_train, X_test, y_train, y_test, meta = preprocess_split(df)
    _, _, _, p_logit = part1_models_and_thresholds(X_train, X_test, y_train, y_test)
    part2_1_age_subgroup_analysis(df)
    part2_2_omitted_variable_bias(df)
    part2_3_adversarial_subset(p_logit, y_test, meta)
    part2_4_fairness_tradeoff(p_logit, y_test, meta)


if __name__ == "__main__":
    main()