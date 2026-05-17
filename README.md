# Trustworthy AI for Insurance Risk Modeling

A **production-style Streamlit application** for auto insurance **claim prediction** using a **Random Forest**, combined with **bias detection**, **fairness metrics**, and **mitigation** (reweighing and group-specific thresholds). The goal is to help technical and non-technical users (e.g. analysts, managers) understand not only *how accurate* a model is, but also *whether it treats customer groups equitably*—and what trade-offs appear when you try to fix unfair patterns.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Who should use it](#who-should-use-it)
3. [Quick start](#quick-start)
4. [Project structure](#project-structure)
5. [The dataset](#the-dataset)
6. [How data is prepared](#how-data-is-prepared)
7. [How the model works](#how-the-model-works)
8. [Train / validation / test splits](#train--validation--test-splits)
9. [Streamlit app: pages and workflow](#streamlit-app-pages-and-workflow)
10. [Sidebar controls explained](#sidebar-controls-explained)
11. [Metrics glossary](#metrics-glossary)
12. [Fairness & bias concepts](#fairness--bias-concepts)
13. [Mitigation strategies](#mitigation-strategies)
14. [FAQ and troubleshooting](#faq-and-troubleshooting)
15. [Limitations and responsible use](#limitations-and-responsible-use)
16. [Extending the project](#extending-the-project)
17. [References](#references)

---

## What this project does

| Capability | Description |
|------------|-------------|
| **Binary classification** | Predict whether a policy will have a claim (`claim_status`: 0 = no, 1 = yes). |
| **Random Forest** | Main model; hyperparameters adjustable in the UI; `class_weight=balanced` for rare claims. |
| **Performance evaluation** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, calibration, feature importance, optional SHAP. |
| **Bias / fairness analysis** | Group-wise selection rate, TPR, FPR; statistical parity, disparate impact, equal opportunity, equalized odds. |
| **Mitigation** | Sample reweighing (retrain) and per-group probability thresholds (post-processing). |
| **Interactive UI** | Six pages with plain-language descriptions under headings and charts. |

This is **not** the classic Kaggle “insurance fraud” dataset (`fraud_reported`, `insured_sex`). It is a **vehicle / policy features** dataset with roughly **58,600 policies** and a **~6.4% claim rate**.

---

## Who should use it

- **Business / insurance users:** Explore claim rates by region or segment, read fairness summaries, compare before/after mitigation without writing code.
- **Data scientists / students:** See an end-to-end sklearn + Fairlearn + Streamlit pipeline with reproducible splits and saved models.
- **Reviewers / instructors:** Use the README and in-app text as documentation for design choices and limitations.

---

## Quick start

### Prerequisites

- **Python 3.10+** (project tested with Python 3.13 in the `HM` venv)
- Dataset file at `data/insurance_claims.csv`

If the file is missing, copy it from the original folder:

```bash
cd Insurance_Risk_Modelling
mkdir -p data
cp "Dataset/Insurance claims data.csv" data/insurance_claims.csv
```

### Install dependencies (HM virtual environment)

This repo includes a local venv named **`HM`**:

```bash
cd Insurance_Risk_Modelling
source HM/bin/activate          # Windows: HM\Scripts\activate
pip install -r requirements.txt
```

To create `HM` from scratch instead:

```bash
python3 -m venv HM
source HM/bin/activate
pip install -r requirements.txt
```

### Run the app

```bash
source HM/bin/activate
streamlit run app.py
```

The app opens in your browser (default `http://localhost:8501`). Use the **sidebar** to switch pages and change model / fairness settings.

### Recommended first-time workflow

1. **Overview** — read context and model card.  
2. **Data Exploration** — check claim rate and differences by `region_code` or `age_group`.  
3. **Model Training & Performance** — leave default features, click **Train Random Forest**, review metrics and charts.  
4. **Fairness & Bias** — pick a sensitive attribute in the sidebar; review group table and charts.  
5. **Mitigation / Constraints** — try reweighing or group thresholds; read the before/after table.  
6. **Single Prediction** — enter one policy and get a probability.

Training with default `n_estimators=200` may take **1–3 minutes** on a laptop.

---

## Project structure

```
Insurance_Risk_Modelling/
├── app.py                 # Streamlit entry point (all UI pages)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   └── insurance_claims.csv
├── Dataset/               # Original CSV (optional backup)
├── models/
│   └── rf_baseline.joblib # Created after training (gitignored)
├── HM/                    # Local virtual environment (gitignored)
└── src/
    ├── config.py          # Paths, target name, split sizes, sensitive attrs
    ├── data_loader.py     # Load CSV, feature engineering, train/val/test split
    ├── modeling.py        # Random Forest pipeline, metrics, save/load
    ├── fairness.py        # Fairness metrics + reweighing + thresholds
    └── plots.py           # Plotly charts + SHAP matplotlib figure
```

### Module responsibilities

| Module | Role |
|--------|------|
| `config.py` | Single source of truth: `DATA_PATH`, `TARGET_COL = "claim_status"`, `RANDOM_STATE = 42`, test/val fractions, sensitive attribute list, age bins. |
| `data_loader.py` | `load_raw()`, `engineer_features()`, `split_data()`, `build_preprocessor()` (numeric scale + categorical one-hot). |
| `modeling.py` | `build_rf_pipeline()`, `train_model()`, `evaluate()`, `predict_proba()`, `save_model()` / `load_model()`. |
| `fairness.py` | Group metrics, Fairlearn + manual fairness summaries, `reweighing_weights()`, `tune_group_thresholds()`. |
| `plots.py` | All chart builders used by `app.py`. |
| `app.py` | Navigation, session state, page layouts, user-facing descriptions. |

---

## The dataset

### Source and location

- **File used by the app:** `data/insurance_claims.csv`  
- **Original copy:** `Dataset/Insurance claims data.csv`  
- **Rows:** ~58,592 policies  
- **Claim rate:** ~6.4% (`claim_status = 1`)

### Target variable

| Column | Meaning |
|--------|---------|
| `claim_status` | **1** = claim occurred, **0** = no claim. This is what the model predicts. |

`policy_id` is **dropped** before modeling (identifier only).

### Feature groups (conceptual)

| Group | Examples |
|-------|----------|
| Policy / customer | `subscription_length`, `customer_age`, `vehicle_age` |
| Geography | `region_code`, `region_density` |
| Vehicle segment | `segment`, `fuel_type`, `engine_type`, `ncap_rating` |
| Engine / performance | `max_torque`, `max_power`, `displacement`, `cylinder` (torque/power parsed to numbers) |
| Safety / equipment | `airbags`, `is_esc`, `is_tpms`, … (Yes/No → 1/0) |
| Dimensions | `length`, `width`, `gross_weight`, `turning_radius` |

High-cardinality `model` is replaced by **`model_freq`** (training-set frequency of that model name) and the raw `model` column is removed.

### Sensitive attributes (for fairness only)

These are used to **slice** the population for fairness analysis—not necessarily removed from training:

| Attribute | Description |
|-----------|-------------|
| `region_code` | Geographic region (rare regions with &lt; 200 policies grouped as `"Other"`) |
| `segment` | Vehicle segment (e.g. A, B1, B2, C1, C2, Utility) |
| `age_group` | Derived from `customer_age`: 35–45, 46–55, 56–65, 66–75 |
| `fuel_type` | e.g. Diesel, CNG, Petrol |

There is **no gender column** in this dataset. Fairness is analyzed along region, segment, age band, and fuel type.

---

## How data is prepared

All preprocessing is implemented in `src/data_loader.py`.

### Step-by-step

1. **Load CSV** from `data/insurance_claims.csv`.  
2. **Parse text numerics:** `max_torque` and `max_power` → first number in the string (e.g. `"250Nm@2750rpm"` → `250`).  
3. **Create `age_group`:** `pd.cut` on `customer_age` with bins `[34, 45, 55, 65, 76]`.  
4. **Encode Yes/No columns** as 1 / 0 for any column whose values are only Yes/No.  
5. **Bucket rare regions:** regions with fewer than **200** policies → `"Other"`.  
6. **Replace `model`** with `model_freq` and drop `model`.  
7. **Split** stratified by `claim_status` into train / validation / test (see below).  
8. **Preprocessor (inside sklearn Pipeline):**  
   - Numeric columns → median imputation + standard scaling  
   - Categorical columns → most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`)

### What is excluded from model features

- `policy_id`  
- `claim_status` (target)  
- `age_group` is **excluded from default training features** (still used as sensitive attribute for fairness). `customer_age` **is** included.

You can change which columns are used on the **Model Training** page via the **Features for training** multiselect.

---

## How the model works

### Algorithm

- **RandomForestClassifier** (scikit-learn) inside a **Pipeline**:  
  `preprocessor` → `clf`  
- Default **`class_weight="balanced"`** so rare claims (6.4%) get higher weight in tree building.  
- **Random seed:** `42` everywhere splits and forest are concerned.

### Hyperparameters (sidebar)

| Parameter | Effect (simple) |
|-----------|------------------|
| `n_estimators` | Number of trees; more → often more stable, slower. |
| `max_depth` | Max tree depth; `None` = grow until pure/leaves limit; lower → less overfitting. |
| `min_samples_split` | Minimum samples to split a node. |
| `min_samples_leaf` | Minimum samples per leaf. |

### Outputs

- **Probability:** `predict_proba`[:, 1] = estimated P(claim).  
- **Class:** probability ≥ **decision threshold** (sidebar, default 0.5) → predict claim (1).

### Saved artifact

After training, the full pipeline is saved to:

`models/rf_baseline.joblib`

Reload in a new Python session with `src.modeling.load_model()`.

---

## Train / validation / test splits

Configured in `src/config.py`:

| Split | Share of full data | Purpose |
|-------|-------------------|---------|
| **Train** | 70% | Fit preprocessor + Random Forest |
| **Validation** | 15% | Metrics in UI (fairness, mitigation tuning), model selection |
| **Test** | 15% | Hold-out evaluation via dropdown on training page |

Splits are **stratified** on `claim_status` so each set has a similar claim rate. The **sensitive attribute** series is split in parallel so each row keeps its group label.

---

## Streamlit app: pages and workflow

Each page includes **short captions** under headings and charts (plain language). Below is a deeper reference.

### 1. Overview

**Purpose:** Context for “trustworthy AI” without running a model.

**Contents:**

- Why historical insurance data can be biased.  
- Why removing sensitive columns is insufficient (proxy features).  
- How fairness metrics and mitigation help.  
- **Model card:** data source, target, sensitive attributes, model type, intended use, limitations.

**No training required.**

---

### 2. Data Exploration

**Purpose:** Understand the data **before** modeling.

| Section | What you see |
|---------|----------------|
| Summary metrics | Row count, feature count, overall claim rate, missing values |
| Sample data | First 200 rows |
| Outcome by sensitive group | Bar chart of **actual** claim rate per group (sidebar sensitive attribute) |
| Feature distribution | Histogram or bar chart for any column you select |
| Correlation heatmap | Numeric feature correlations |

**Insight:** If claim rates differ a lot by region or age in the **raw data**, a model may learn—and amplify—those patterns.

---

### 3. Model Training & Performance

**Purpose:** Train the forest and evaluate discrimination ability.

| Section | What you see |
|---------|----------------|
| Feature multiselect | Choose inputs (default: all engineered features except `policy_id`, target, `age_group`) |
| Train button | Fits pipeline; saves `models/rf_baseline.joblib`; stores model in session |
| Summary metrics | Accuracy, precision, recall, F1, ROC-AUC |
| ROC curve | Trade-off true positive vs false positive rate; **AUC** summarized |
| Confusion matrix | Counts: true/false positives and negatives |
| Calibration plot | Whether predicted probabilities match actual claim frequencies |
| Probability distribution | Scores for policies that did vs did not claim |
| Feature importance | Global Random Forest importances (after preprocessing) |
| SHAP (optional) | Directional impact of features on a sample of policies (slow) |

**Evaluation split:** Validation (default) or Test.

**Sidebar mitigation on train:** If **Reweighing** is selected in the sidebar before training, sample weights are applied during that train click.

---

### 4. Fairness & Bias

**Purpose:** Detect **unequal model behavior** across groups.

**Requires:** A trained model.

**Uses:** Validation set predictions and the sidebar **sensitive attribute**.

| Section | Meaning |
|---------|---------|
| Group metrics table | Per group: count, selection rate, TPR, FPR, prevalence |
| Charts (tabs) | Bar charts for selection rate, TPR, FPR by group |
| Fairness metrics | Single-number summaries (see [Fairness & bias concepts](#fairness--bias-concepts)) |

**Important:** This page measures **model outcomes**, not legal “discrimination.” It flags **disparities** worth investigating.

---

### 5. Mitigation / Constraints

**Purpose:** Try to **reduce** fairness gaps; show **accuracy vs fairness** trade-offs.

**Requires:** A trained model.

| Option | Type | What happens |
|--------|------|----------------|
| **None** | — | Baseline: everyone uses probability ≥ 0.5 |
| **Reweighing (retrain)** | Pre-processing | Click button to retrain with AIF360-style sample weights on (group, label) cells |
| **Group threshold tuning** | Post-processing | Different probability cutoffs per group; scores unchanged |

| Section | Meaning |
|---------|---------|
| Before vs after table | Accuracy and fairness metrics: baseline (0.5 cutoff) vs after mitigation |
| Trade-off scatter plots | Sweep **one global** threshold from 0.05 to 0.95; each point = accuracy vs a fairness metric |

**Note:** Group threshold JSON shows one cutoff per region/segment/etc. Lower cutoff → more policies predicted as claims for that group.

---

### 6. Single Prediction

**Purpose:** What-if scoring for **one policy**.

- Key inputs (age, region, segment, …) + expandable additional features.  
- Output: **probability of claim** and **class** (uses sidebar decision threshold).  
- Table of **global** top feature importances (not a full SHAP explanation for that row).

---

## Sidebar controls explained

| Control | Applies to |
|---------|------------|
| **Navigate** | All pages |
| **n_estimators, max_depth, min_samples_split, min_samples_leaf** | Training only |
| **Decision threshold** | Turns probability into class; affects fairness page predictions, single prediction, and metric display on training page |
| **Sensitive attribute** | Fairness & Bias, Mitigation (and reweighing uses training sensitive labels) |
| **Mitigation (training page)** | `None` or `Reweighing` when you click **Train** on page 3 |

---

## Metrics glossary

### Classification (Model Training page)

| Metric | Question it answers | Good direction |
|--------|----------------------|----------------|
| **Accuracy** | What fraction of all predictions are correct? | Higher |
| **Precision** | When we predict “claim”, how often is that true? | Higher (fewer false alarms) |
| **Recall** | Of all real claims, what fraction did we catch? | Higher (fewer missed claims) |
| **F1** | Balance of precision and recall | Higher |
| **ROC-AUC** | How well does the model **rank** risky policies vs safe ones? | Higher (1.0 = perfect ranking) |

With **imbalanced** data (~6% claims), accuracy can look high while recall is poor—always read precision and recall together.

### Group-level (Fairness page)

| Metric | Definition (plain language) |
|--------|----------------------------|
| **Selection rate** | % of policies in the group predicted as “claim” |
| **TPR (true positive rate)** | Among policies that **actually had** a claim, % correctly predicted as claim |
| **FPR (false positive rate)** | Among policies with **no** claim, % wrongly predicted as claim |
| **Prevalence** | Actual claim rate in that group in the data (not the model) |

### Fairness summaries (single numbers across groups)

| Metric | Plain language | Closer to “fair” |
|--------|----------------|------------------|
| **Statistical parity difference** | Gap between highest and lowest **selection rate** | **0** |
| **Disparate impact** (ratio) | Min selection rate ÷ max selection rate | **1.0** |
| **Equal opportunity difference** | Gap between highest and lowest **TPR** (among real claimants) | **0** |
| **Equalized odds difference** | Worst gap in TPR **or** FPR across groups (Fairlearn) | **0** |

Fairlearn is used when installed; manual fallbacks exist in `src/fairness.py` if not.

---

## Fairness & bias concepts

### What is “bias” here?

We mean **unequal model behavior** across a sensitive group (e.g. regions):

- Flagging claims more often in some regions (**selection rate**).  
- Missing real claims more often in some age bands (**low TPR**).  
- More false alarms in some segments (**high FPR**).

This can come from **historical data**, **class imbalance**, or **correlated features** (proxies)—not only from using region/age directly.

### Why not just drop `region_code`?

Other columns (density, segment, vehicle type) may **correlate** with region. The model can still produce disparate outcomes. Fairness analysis makes that visible.

### Relationship between pages

```
Data Exploration     →  “Do claim rates differ by group in history?”
Model Training       →  “How accurate is the model overall?”
Fairness & Bias      →  “Does the model treat groups differently?”
Mitigation           →  “Can we reduce gaps—and at what accuracy cost?”
```

---

## Mitigation strategies

### 1. Reweighing (pre-processing)

**File:** `reweighing_weights()` in `src/fairness.py`

- For each combination of **sensitive group** and **label** (claim / no claim), assign a weight inversely related to how common that cell is.  
- Retrain Random Forest with `sample_weight` on training data.  
- Predictions still use a **single** threshold (0.5) unless you change it.

**Pros:** Can improve parity-type metrics at the source.  
**Cons:** Retraining cost; accuracy may drop; weights computed on training set only.

### 2. Group threshold tuning (post-processing)

**Files:** `tune_group_thresholds()`, `apply_group_thresholds()`

- Model **scores unchanged**; each group gets its own cutoff on probability.  
- Tuning uses **coordinate descent** on a grid (0.1–0.9) to reduce **equal opportunity difference** on the validation set.

**Pros:** Fast; no retrain.  
**Cons:** Same score can yield different decisions by group—may be unacceptable in production; policy/legal review needed.

### 3. Global threshold sweep (charts only)

The trade-off plots vary **one** threshold for everyone. They illustrate that fairness and accuracy often **move together** along the curve—they are not separate mitigation buttons.

---

## FAQ and troubleshooting

### The app says “Dataset not found”

Ensure `data/insurance_claims.csv` exists:

```bash
cp "Dataset/Insurance claims data.csv" data/insurance_claims.csv
```

### “Train a model first” on Fairness / Mitigation / Prediction

Go to **Model Training & Performance** and click **Train Random Forest**. Wait until the success message appears.

### Training is very slow

- Reduce **n_estimators** (e.g. 50–100) in the sidebar.  
- Uncheck SHAP unless needed.  
- Close other heavy applications.

### ROC-AUC seems modest (~0.65)

Claims are rare and signal may be weak; this is common. Focus on **recall vs precision** for claim detection and on **fairness gaps**, not only AUC.

### Fairness numbers look extreme (e.g. disparate impact near 0)

With **many groups** (regions) and **imbalanced** data, some groups have few claimants—TPR/FPR become unstable. Prefer **`age_group`** or **`segment`** for stabler charts, or interpret small groups cautiously.

### SHAP does not show

Install `shap` in the active environment (`pip install shap`). SHAP needs enough RAM; try fewer trees or a smaller training sample (code uses up to 500 rows).

### Reweighing button does nothing visible until you compare

After retrain, check **Mitigation → Before vs after** or revisit **Fairness & Bias** with the new model in session.

### ModuleNotFoundError for `src`

Run Streamlit from the **project root** (`Insurance_Risk_Modelling`), not from inside `src/`:

```bash
cd Insurance_Risk_Modelling
streamlit run app.py
```

### Wrong Python / packages

Confirm the HM venv is active (`which python` should point to `.../HM/bin/python`) and run `pip install -r requirements.txt` again.

### Session state lost

Refreshing the browser may clear the trained model in memory. **Re-train** or load from `models/rf_baseline.joblib` via Python (loading in the UI is not exposed as a button by default).

---

## Limitations and responsible use

| Topic | Limitation |
|-------|------------|
| **Not legal advice** | Fairness metrics are **technical diagnostics**, not compliance with any insurance regulation. |
| **Validation-only mitigation tuning** | Group thresholds are fit on **validation** data; test set should be used for final reporting to avoid overfitting mitigation. |
| **Geography / product scope** | Model trained on one dataset; may not transfer to other countries, products, or time periods. |
| **No causal fairness** | Observed disparities do not prove intent or identify root cause. |
| **Production deployment** | No API, auth, logging, or monitoring—this is an **analysis prototype**. |
| **Single sensitive attribute** | UI analyzes one attribute at a time, not intersections (e.g. region × age). |

**Intended use:** Education, exploratory analysis, and demonstrations of trustworthy-AI workflows—not automated underwriting decisions without human oversight and domain validation.

---

## Extending the project

| Goal | Where to change |
|------|----------------|
| New sensitive attribute | Add to `SENSITIVE_CANDIDATES` in `src/config.py`; ensure column exists after `engineer_features()`. |
| Different target column | Change `TARGET_COL` in `config.py` and re-engineer labels. |
| Another classifier | Replace `RandomForestClassifier` in `src/modeling.py`; keep Pipeline structure. |
| Fairlearn `ThresholdOptimizer` | Integrate in `src/fairness.py` and wire a new option in `app.py` mitigation page. |
| Load model on startup | Call `load_model()` in `app.py` `init_session_state` if file exists. |
| More pages | Add to sidebar radio and `pages` dict in `main()`. |

---

## References

- [Fairlearn documentation](https://fairlearn.org/) — metrics and mitigation concepts  
- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)  
- [SHAP](https://shap.readthedocs.io/) — explainability  
- AIF360 reweighing idea — implemented manually in `reweighing_weights()`

---

## Dependencies (`requirements.txt`)

| Package | Role |
|---------|------|
| streamlit | Web UI |
| pandas, numpy | Data |
| scikit-learn | Model, preprocessing, metrics |
| plotly, matplotlib, seaborn | Charts |
| joblib | Model persistence |
| fairlearn | Fairness metrics |
| shap | Optional SHAP plots |

---

## Summary

This project answers three questions in one app:

1. **How well can we predict claims?** → Model Training page.  
2. **Does the model treat groups fairly?** → Fairness & Bias page.  
3. **Can we improve fairness, and what do we give up?** → Mitigation page.

If you read this README and follow the recommended workflow, you should be able to run the app, interpret every major chart and metric, and explain design choices to technical and non-technical stakeholders—without reading the source code first.

For in-app short hints, each page also shows captions under headings and charts when you run `streamlit run app.py`.
