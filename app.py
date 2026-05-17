"""
Trustworthy AI for Insurance Risk Modeling — Streamlit application.
Run: streamlit run app.py  (from project root, with HM venv activated)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, SENSITIVE_CANDIDATES, TARGET_COL
from src.data_loader import (
    default_feature_columns,
    engineer_features,
    load_and_prepare,
    load_raw,
)
from src.fairness import (
    FAIRNESS_EXPLANATIONS,
    apply_group_thresholds,
    fairness_summary,
    group_metrics_table,
    mitigation_comparison,
    reweighing_weights,
    tradeoff_curve,
    tune_group_thresholds,
)
from src.modeling import (
    build_rf_pipeline,
    evaluate,
    get_feature_importances,
    predict_proba,
    save_model,
    train_model,
)
from src.plots import (
    plot_calibration,
    plot_claim_rate_by_group,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_group_bars,
    plot_probability_distribution,
    plot_roc,
    plot_tradeoff,
    shap_summary_figure,
)

st.set_page_config(
    page_title="Trustworthy Insurance Risk AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def describe(text: str) -> None:
    """Short plain-language note shown under a heading or chart."""
    st.caption(text)


def init_session_state() -> None:
    defaults = {
        "pipeline": None,
        "mitigation": "None",
        "group_thresholds": None,
        "trained": False,
        "data_bundle": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_data(show_spinner="Loading dataset…")
def cached_engineered_df() -> pd.DataFrame:
    return engineer_features(load_raw())


def sidebar_controls() -> dict:
    """Global sidebar: navigation, RF params, sensitive attribute."""
    st.sidebar.title("Controls")
    describe(
        "Pick a page, tune the forest, set the claim cutoff, and choose which group to use "
        "for fairness checks."
    )
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Data Exploration",
            "Model Training & Performance",
            "Fairness & Bias",
            "Mitigation / Constraints",
            "Single Prediction",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model (Random Forest)")
    st.sidebar.caption(
        "More trees = often more stable but slower. max_depth limits tree size to reduce overfitting."
    )
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, 25)
    max_depth = st.sidebar.selectbox(
        "max_depth",
        options=[None, 5, 10, 15, 20, 30],
        index=0,
        format_func=lambda x: "None (unlimited)" if x is None else str(x),
    )
    min_samples_split = st.sidebar.slider("min_samples_split", 2, 20, 2)
    min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 10, 1)
    decision_threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    st.sidebar.caption(
        "Probabilities at or above this value count as 'claim'. Lower = more claims predicted."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Fairness")
    st.sidebar.caption(
        "Group used on Fairness and Mitigation pages—for example region or age band."
    )
    sensitive_col = st.sidebar.selectbox("Sensitive attribute", SENSITIVE_CANDIDATES)
    mitigation = st.sidebar.selectbox(
        "Mitigation (training page)",
        ["None", "Reweighing", "Group thresholding"],
        key="mitigation_select",
    )

    return {
        "page": page,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "decision_threshold": decision_threshold,
        "sensitive_col": sensitive_col,
        "mitigation": mitigation,
    }


def page_overview() -> None:
    st.title("Trustworthy AI for Insurance Risk Modeling")
    describe(
        "Welcome. This app predicts whether a policy is likely to have a claim, checks if the "
        "model treats customer groups fairly, and lets you try simple fixes when it does not."
    )
    st.markdown(
        """
        This application helps insurance teams **predict claim likelihood**, detect **bias** in model
        behavior across groups, and explore **fairness constraints** that balance accuracy with
        equitable treatment.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Why historical data can be biased")
        describe(
            "Old claims data often mirrors past business rules and regional differences—not always "
            "what we want the model to copy going forward."
        )
        st.markdown(
            """
            Past claims reflect geography (repair costs), vehicle segments, age-related driving
            patterns, and underwriting practices. Models trained on this history can **repeat or
            amplify** those patterns—even when we only want to predict risk.
            """
        )
        st.subheader("Why dropping sensitive fields is not enough")
        describe(
            "If you remove region or age from the form, the model can still infer them from "
            "other columns like density, segment, or vehicle type."
        )
        st.markdown(
            """
            Removing `region_code` or `age_group` does not remove **proxy variables** (density,
            vehicle segment, NCAP rating). The model may still treat groups differently.
            """
        )
    with col2:
        st.subheader("How fairness metrics help")
        describe(
            "Fairness numbers show whether the model flags claims—or misses real claims—at "
            "similar rates for different regions, ages, or segments."
        )
        st.markdown(
            """
            We measure **selection rates**, **true positive rates**, and standard metrics like
            statistical parity difference and equal opportunity difference. **Mitigation** (sample
            reweighing or group-specific thresholds) can reduce disparities at some cost to accuracy.
            """
        )
        st.subheader("Model card")
        describe(
            "A quick summary of what data and model this app uses, and what it should—not—be used for."
        )
        st.markdown(
            f"""
            | Item | Detail |
            |------|--------|
            | **Data source** | Auto insurance claims dataset (`{DATA_PATH.name}`) |
            | **Target** | `{TARGET_COL}` (1 = claim occurred) |
            | **Sensitive attributes** | {", ".join(SENSITIVE_CANDIDATES)} |
            | **Model** | Random Forest classifier (`class_weight=balanced`) |
            | **Intended use** | Exploratory risk scoring and fairness analysis—not sole basis for underwriting |
            | **Limitations** | May not generalize to other regions/products; imbalanced labels (~6% claims) |
            """
        )


def page_data_exploration(df: pd.DataFrame, sensitive_col: str) -> None:
    st.header("Data Exploration")
    describe(
        "Look at the raw data before modeling: how many policies, how often claims happen, "
        "and whether claim rates differ by region, segment, or other groups."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", len(df.columns))
    c3.metric("Claim rate", f"{df[TARGET_COL].mean():.2%}")
    c4.metric("Missing values", int(df.isnull().sum().sum()))
    describe(
        "Rows and features count the dataset size. Claim rate is the share of policies with a "
        "claim. Missing values should be zero after cleaning."
    )

    st.subheader("Sample data")
    describe("First 200 rows so you can see column names, values, and typical policy records.")
    st.dataframe(df.head(200), use_container_width=True)

    st.subheader("Outcome by sensitive group")
    describe(
        f"Real claim rate in the data for each {sensitive_col} group—before any model. "
        "Large gaps here can lead to unfair model behavior later."
    )
    st.plotly_chart(
        plot_claim_rate_by_group(df, sensitive_col, TARGET_COL),
        use_container_width=True,
    )
    describe(
        "Each bar is the actual % of policies with a claim in that group. Taller bars mean "
        "that group had more claims in history—not model predictions."
    )

    col_a, col_b = st.columns(2)
    st.subheader("Feature distributions & correlations")
    describe(
        "Explore one feature at a time, then see how numeric features move together. "
        "Strong correlations can mean redundant inputs for the model."
    )
    explore_col = col_a.selectbox("Feature distribution", df.columns.tolist(), index=3)
    col_a.plotly_chart(plot_distribution(df, explore_col), use_container_width=True)
    col_a.caption(
        "Histogram for numbers or bar chart for categories. Helps spot skew, rare values, "
        "and whether a feature looks useful for prediction."
    )

    num_df = df.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        col_b.plotly_chart(plot_correlation_heatmap(num_df), use_container_width=True)
        col_b.caption(
            "Colors show how strongly pairs of numeric features increase or decrease together. "
            "Red = positive correlation, blue = negative."
        )


def page_model_training(controls: dict, df: pd.DataFrame) -> None:
    st.header("Model Training & Performance")
    describe(
        "Train a Random Forest to predict claims, then review how accurate it is. "
        "Use the sidebar to change tree settings; pick features below if you want a smaller model."
    )

    st.subheader("Choose features & train")
    describe(
        "Select which columns the model may use. Click **Train Random Forest** when ready—"
        "training can take a minute with many trees."
    )
    all_features = default_feature_columns(df)
    feature_cols = st.multiselect(
        "Features for training",
        all_features,
        default=all_features,
    )
    if not feature_cols:
        st.warning("Select at least one feature.")
        return

    bundle = load_and_prepare(
        sensitive_col=controls["sensitive_col"],
        feature_cols=feature_cols,
    )
    st.session_state["data_bundle"] = bundle

    st.caption(f"Train / val / test: {len(bundle['X_train'])} / {len(bundle['X_val'])} / {len(bundle['X_test'])}")

    if st.button("Train Random Forest", type="primary"):
        with st.spinner("Training…"):
            pipe = build_rf_pipeline(
                bundle["preprocessor"],
                n_estimators=controls["n_estimators"],
                max_depth=controls["max_depth"],
                min_samples_split=controls["min_samples_split"],
                min_samples_leaf=controls["min_samples_leaf"],
            )
            weights = None
            if controls["mitigation"] == "Reweighing":
                weights = reweighing_weights(
                    bundle["y_train"].values,
                    bundle["s_train"].values,
                )
            train_model(pipe, bundle["X_train"], bundle["y_train"], sample_weight=weights)
            save_model(pipe, "rf_baseline.joblib")
            st.session_state["pipeline"] = pipe
            st.session_state["trained"] = True
            st.session_state["mitigation"] = controls["mitigation"]
            st.success("Model trained and saved to `models/rf_baseline.joblib`.")

    pipe = st.session_state.get("pipeline")
    if not pipe or not st.session_state.get("trained"):
        st.info("Configure hyperparameters in the sidebar and click **Train Random Forest**.")
        return

    st.subheader("Model performance")
    describe(
        "Scores and charts for data the model did not train on. Validation is used for tuning; "
        "test is a final check. The decision threshold (sidebar) turns probability into yes/no claim."
    )
    split_name = st.selectbox("Evaluation split", ["Validation", "Test"])
    X = bundle["X_val"] if split_name == "Validation" else bundle["X_test"]
    y = bundle["y_val"] if split_name == "Validation" else bundle["y_test"]

    thr = controls["decision_threshold"]
    metrics = evaluate(pipe, X, y, threshold=thr)

    st.markdown("**Summary metrics**")
    describe(
        "Accuracy = overall correct guesses. Precision = when we predict claim, how often we're right. "
        "Recall = of real claims, how many we catch. F1 balances precision and recall. ROC-AUC = "
        "how well the model ranks risky policies (higher is better, max 1.0)."
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1", f"{metrics['f1']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            plot_roc(metrics["fpr"], metrics["tpr"], metrics["roc_auc"]),
            use_container_width=True,
        )
        describe(
            "ROC curve: trade-off between catching real claims (true positive rate, up) and "
            "false alarms (false positive rate, right). A curve hugging the top-left is better."
        )
        st.plotly_chart(
            plot_confusion_matrix(metrics["confusion_matrix"]),
            use_container_width=True,
        )
        describe(
            "Confusion matrix: counts of correct and wrong predictions. Top-left = true no-claim; "
            "bottom-right = true claim; off-diagonal = mistakes."
        )
    with col_r:
        st.plotly_chart(
            plot_calibration(y, metrics["proba"]),
            use_container_width=True,
        )
        describe(
            "Calibration: if the model says 30% risk, do about 30% of those policies actually claim? "
            "Points near the diagonal mean probabilities are trustworthy."
        )
        st.plotly_chart(
            plot_probability_distribution(metrics["proba"], y),
            use_container_width=True,
        )
        describe(
            "Distribution of predicted claim probabilities for policies that did vs did not claim. "
            "Good separation means the model gives higher scores to real claims."
        )

    st.subheader("What drives predictions")
    describe(
        "Which inputs the forest relied on most. Longer bars mean that feature split the data "
        "more often when building trees."
    )
    try:
        names, imps = get_feature_importances(pipe)
        st.plotly_chart(
            plot_feature_importance(names, imps),
            use_container_width=True,
        )
        describe(
            "Global feature importance across the whole model—not for one policy only. "
            "Names like num__customer_age are engineered columns after preprocessing."
        )
    except Exception as exc:
        st.warning(f"Could not plot feature importances: {exc}")

    st.subheader("SHAP explanation (optional)")
    describe(
        "SHAP shows how each feature pushes risk up or down across many sample policies. "
        "Useful for experts; slower to compute."
    )
    if st.checkbox("Compute SHAP summary (slower)"):
        sample = bundle["X_train"].sample(
            min(500, len(bundle["X_train"])),
            random_state=42,
        )
        fig = shap_summary_figure(pipe, sample)
        if fig is not None:
            st.pyplot(fig)
            describe(
                "Each dot is a policy. Color = feature value; horizontal position = impact on "
                "claim score. Features are sorted by overall influence."
            )
        else:
            st.warning("SHAP plot unavailable. Check that shap is installed.")


def _get_eval_bundle(controls: dict, df: pd.DataFrame):
    bundle = st.session_state.get("data_bundle")
    if bundle is None or bundle.get("sensitive_col") != controls["sensitive_col"]:
        bundle = load_and_prepare(
            sensitive_col=controls["sensitive_col"],
            feature_cols=default_feature_columns(df),
        )
        st.session_state["data_bundle"] = bundle
    return bundle


def page_fairness(controls: dict, df: pd.DataFrame) -> None:
    st.header("Fairness & Bias")
    describe(
        "Checks whether the trained model treats groups differently—for example, flagging more "
        "claims in some regions than others. Change the sensitive attribute in the sidebar."
    )
    pipe = st.session_state.get("pipeline")
    if not pipe or not st.session_state.get("trained"):
        st.warning("Train a model on **Model Training & Performance** first.")
        return

    bundle = _get_eval_bundle(controls, df)
    y = bundle["y_val"].values
    sensitive = bundle["s_val"]
    proba = predict_proba(pipe, bundle["X_val"])
    pred = (proba >= controls["decision_threshold"]).astype(int)

    st.subheader(f"Group metrics — sensitive: `{controls['sensitive_col']}`")
    describe(
        "One row per group on validation data. Compare selection rate, TPR, and FPR across rows—"
        "big gaps suggest bias in model outcomes."
    )
    gdf = group_metrics_table(y, pred, sensitive)
    st.dataframe(
        gdf.style.format(
            {
                "selection_rate": "{:.3f}",
                "tpr": "{:.3f}",
                "fpr": "{:.3f}",
                "prevalence": "{:.3f}",
            }
        ),
        use_container_width=True,
    )
    describe(
        "Selection rate = % predicted as claim. TPR = % of real claims the model caught. "
        "FPR = % of no-claim policies wrongly flagged. Prevalence = actual claim % in that group."
    )

    st.subheader("Charts by group")
    describe("Same table as bar charts—easier to spot which group stands out.")
    tab1, tab2, tab3 = st.tabs(["Selection rate", "TPR", "FPR"])
    with tab1:
        describe("Share of each group the model labels as 'claim'—should be similar if treatment is fair.")
        st.plotly_chart(
            plot_group_bars(gdf, "selection_rate", "Selection rate by group"),
            use_container_width=True,
        )
    with tab2:
        describe(
            "Among policies that really had a claim, how often did the model say 'claim'? "
            "Low bars mean the model misses claims in that group."
        )
        st.plotly_chart(
            plot_group_bars(gdf, "tpr", "True positive rate by group"),
            use_container_width=True,
        )
    with tab3:
        describe(
            "Among policies with no claim, how often did the model still predict 'claim'? "
            "High bars mean extra false alarms for that group."
        )
        st.plotly_chart(
            plot_group_bars(gdf, "fpr", "False positive rate by group"),
            use_container_width=True,
        )

    st.subheader("Fairness metrics (validation set)")
    describe(
        "Single numbers summarizing inequality across all groups. Closer to zero (or impact ratio "
        "near 1.0) usually means fairer predictions."
    )
    summary = fairness_summary(y, pred, sensitive)
    for metric, value in summary.items():
        st.metric(metric.replace("_", " ").title(), f"{value:.4f}")
        expl = FAIRNESS_EXPLANATIONS.get(metric)
        if expl:
            st.caption(expl)


def page_mitigation(controls: dict, df: pd.DataFrame) -> None:
    st.header("Mitigation / Constraints")
    describe(
        "Try to reduce unfair gaps between groups. You may give up some accuracy—that trade-off "
        "is normal and is shown in the tables and charts below."
    )
    pipe = st.session_state.get("pipeline")
    if not pipe or not st.session_state.get("trained"):
        st.warning("Train a model first.")
        return

    bundle = _get_eval_bundle(controls, df)
    y = bundle["y_val"].values
    sensitive = bundle["s_val"]
    proba = predict_proba(pipe, bundle["X_val"])

    st.subheader("Choose a mitigation strategy")
    describe(
        "**None** = standard 0.5 cutoff. **Reweighing** = retrain with adjusted sample weights. "
        "**Group threshold tuning** = different probability cutoffs per group, same scores."
    )
    mitigation = st.selectbox(
        "Apply mitigation",
        ["None", "Reweighing (retrain)", "Group threshold tuning"],
    )

    baseline_pred = (proba >= 0.5).astype(int)

    if mitigation == "Reweighing (retrain)":
        describe(
            "Gives more weight to underrepresented group-and-outcome combinations during training "
            "so the forest is less dominated by majority patterns."
        )
        if st.button("Retrain with reweighing"):
            with st.spinner("Retraining with sample reweighing…"):
                pipe_rw = build_rf_pipeline(bundle["preprocessor"])
                w = reweighing_weights(
                    bundle["y_train"].values,
                    bundle["s_train"].values,
                )
                train_model(pipe_rw, bundle["X_train"], bundle["y_train"], sample_weight=w)
                st.session_state["pipeline"] = pipe_rw
                st.session_state["mitigation"] = "Reweighing"
                pipe = pipe_rw
                proba = predict_proba(pipe, bundle["X_val"])
                baseline_pred = (proba >= 0.5).astype(int)
            st.success("Retrained with reweighing weights.")
        mitigated_pred = (proba >= 0.5).astype(int)
    elif mitigation == "Group threshold tuning":
        describe(
            "Keeps the same risk scores but changes the 'approve claim' cutoff per group—for "
            "example, stricter in one region, looser in another—to balance true positive rates."
        )
        thresholds = tune_group_thresholds(y, proba, sensitive)
        st.session_state["group_thresholds"] = thresholds
        mitigated_pred = apply_group_thresholds(proba, sensitive, thresholds)
        st.json(thresholds)
        describe(
            "Each number is the minimum probability needed in that group to predict 'claim'. "
            "Lower threshold = more policies flagged as claims."
        )
    else:
        mitigated_pred = baseline_pred
        describe("No change applied—baseline and after rows will match.")

    comp = mitigation_comparison(y, proba, sensitive, baseline_pred, mitigated_pred)
    st.subheader("Before vs after")
    describe(
        "Compare accuracy and fairness before and after your chosen fix. Look for smaller "
        "fairness gaps in the second row; check whether accuracy dropped."
    )
    st.dataframe(
        comp.style.format(
            {c: "{:.4f}" for c in comp.columns if c != "model"},
        ),
        use_container_width=True,
    )

    st.subheader("Accuracy vs fairness trade-off (threshold sweep)")
    describe(
        "Uses one global cutoff for everyone (not per-group). Each point is a different cutoff "
        "from 0.05 to 0.95—helps you see you often cannot improve fairness without hurting accuracy."
    )
    curve = tradeoff_curve(y, proba, sensitive)
    st.plotly_chart(
        plot_tradeoff(
            curve,
            x="accuracy",
            y="equal_opportunity_difference",
            title="Accuracy vs equal opportunity difference",
        ),
        use_container_width=True,
    )
    describe(
        "Left usually means more correct overall; lower on the chart means fairer recall across "
        "groups. Hover to see which threshold was used."
    )
    st.plotly_chart(
        plot_tradeoff(
            curve,
            x="accuracy",
            y="statistical_parity_difference",
            title="Accuracy vs statistical parity difference",
        ),
        use_container_width=True,
    )
    describe(
        "Same idea for flagging rates: moving left may improve accuracy while moving down reduces "
        "how differently groups are marked as 'claim'."
    )


def page_prediction(controls: dict, df: pd.DataFrame) -> None:
    st.header("Single Prediction")
    describe(
        "Enter details for one policy and get a claim probability and yes/no decision. "
        "Useful for what-if checks—not a replacement for full portfolio scoring."
    )
    pipe = st.session_state.get("pipeline")
    if not pipe or not st.session_state.get("trained"):
        st.warning("Train a model first.")
        return

    bundle = st.session_state.get("data_bundle")
    feature_cols = bundle["feature_cols"] if bundle else default_feature_columns(df)

    priority = [
        "customer_age",
        "vehicle_age",
        "subscription_length",
        "region_code",
        "segment",
        "fuel_type",
        "region_density",
        "ncap_rating",
    ]
    primary = [c for c in priority if c in feature_cols]
    secondary = [c for c in feature_cols if c not in primary]

    def _input_widget(container, col: str, row: pd.Series) -> Any:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            lo = float(df[col].min())
            hi = float(df[col].max())
            val = float(row[col]) if pd.notna(row[col]) else lo
            return container.number_input(col, min_value=lo, max_value=hi, value=val)
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        default = str(row[col]) if pd.notna(row[col]) else options[0]
        idx = options.index(default) if default in options else 0
        return container.selectbox(col, options, index=idx)

    with st.form("predict_form"):
        inputs = {}
        row = df.iloc[0]
        st.subheader("Key inputs")
        describe("Main policy fields that usually matter most for risk.")
        cols = st.columns(3)
        for i, col in enumerate(primary):
            inputs[col] = _input_widget(cols[i % 3], col, row)

        with st.expander("Additional features"):
            describe("Optional vehicle and safety details—defaults match a sample policy.")
            cols2 = st.columns(3)
            for i, col in enumerate(secondary):
                inputs[col] = _input_widget(cols2[i % 3], col, row)

        submitted = st.form_submit_button("Predict")

    if submitted:
        st.subheader("Prediction result")
        describe(
            "Probability = model’s estimated chance of a claim. Class uses the sidebar threshold "
            "to turn that into claim / no claim."
        )
        X_one = pd.DataFrame([inputs])[feature_cols]
        proba = float(predict_proba(pipe, X_one)[0])
        pred = int(proba >= controls["decision_threshold"])
        st.success(f"**Predicted probability of claim:** {proba:.2%}")
        st.info(f"**Predicted class** (threshold={controls['decision_threshold']:.2f}): "
                f"{'Claim' if pred == 1 else 'No claim'}")

        try:
            names, imps = get_feature_importances(pipe)
            top = pd.DataFrame({"feature": names[:10], "importance": imps[:10]})
            st.subheader("Top global features (model-wide importances)")
            describe(
                "These are the most influential inputs for the whole model—not a breakdown for "
                "this policy only. For per-policy reasons, use SHAP on the training page."
            )
            st.dataframe(top, use_container_width=True)
        except Exception:
            pass


def main() -> None:
    init_session_state()
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Place `insurance_claims.csv` in `data/`.")
        st.stop()

    controls = sidebar_controls()
    df = cached_engineered_df()

    pages = {
        "Overview": page_overview,
        "Data Exploration": lambda: page_data_exploration(df, controls["sensitive_col"]),
        "Model Training & Performance": lambda: page_model_training(controls, df),
        "Fairness & Bias": lambda: page_fairness(controls, df),
        "Mitigation / Constraints": lambda: page_mitigation(controls, df),
        "Single Prediction": lambda: page_prediction(controls, df),
    }
    pages[controls["page"]]()


if __name__ == "__main__":
    main()
