# Documentation — Cost-Aware Fraud Detection

Full technical walkthrough of the methodology, decisions, and architecture. See `README.md` for the quick overview.

---

## 1. Problem framing

284,807 transactions, 492 fraud cases (0.173%). The central challenge of this project isn't model accuracy — a trivial model that predicts "not fraud" for everything scores 99.83% accuracy while catching zero fraud. The actual problems solved here:

1. How do you validate a model honestly on data this imbalanced?
2. How do you pick a decision threshold that reflects a real business tradeoff, not an arbitrary 0.5?
3. How do you know if the model's confidence scores are trustworthy?
4. Can an unsupervised signal add anything on top of supervised learning here?

---

## 2. Data pipeline

### 2.1 Features
- `V1`–`V28`: PCA-transformed components (original features withheld for confidentiality by the dataset provider)
- `Time`: seconds elapsed since the first transaction in the dataset
- `Amount`: transaction amount
- `Class`: target (1 = fraud, 0 = legit)

### 2.2 Preprocessing decisions
- `Amount` and `Time` are the only unscaled features (V1–V28 arrive already roughly standardized from PCA) — both are scaled with `StandardScaler`.
- **Critical detail**: the scaler is fit on the training set only (`scaler.fit_transform(train)` then `scaler.transform(test)`), never on the full dataset before splitting. Fitting on the full dataset first would leak test-set statistics into training — a common, easy-to-miss mistake in imbalanced/time-series-adjacent problems.

### 2.3 Train/test split — time-based, not random
The dataset spans roughly two days of real transactions in chronological order (`Time` column). Two split strategies were tested:

| Split type | PR-AUC | Why |
|---|---|---|
| Random 80/20, stratified | 0.86 | Near-duplicate/similar transactions can land in both train and test, inflating the score |
| Time-based (first 80% train, last 20% test) | 0.80 | Simulates the real deployment scenario: train on the past, predict the future |

**The time-based split is the one reported as the final result.** The random-split number is documented explicitly as the less honest of the two, not hidden — this is a deliberate methodology choice, not an error correction after the fact.

---

## 3. Model

### 3.1 Why XGBoost
Handles tabular data with mixed feature scales well, supports `scale_pos_weight` for built-in class imbalance handling (rather than requiring external resampling like SMOTE), and integrates cleanly with SHAP for explainability.

### 3.2 Imbalance handling: `scale_pos_weight`
```
scale_pos_weight = (# negative training examples) / (# positive training examples) ≈ 545
```
This tells XGBoost to weight a missed fraud case ~545x more heavily than a misclassified legit transaction during training, without altering the underlying data distribution (unlike oversampling/undersampling).

### 3.3 Hybrid feature: Isolation Forest anomaly score
An `IsolationForest` is trained **unsupervised** (no `Class` labels) on the same feature set. Its `score_samples()` output — sign-flipped so higher = more anomalous — is added as an extra column (`anomaly_score`) to the feature set XGBoost trains on.

**Rationale**: gives the supervised model a "how structurally unusual is this transaction" signal that isn't derived from having seen labeled fraud examples like it before — potentially useful for catching fraud patterns that don't closely resemble the ~400 fraud examples in the training set.

**Result**: `anomaly_score` did not rank in the top 5 features by XGBoost's gain metric (V14, V4, V8, V10, V20 dominate), but ranked **#4 by mean |SHAP value|**. These metrics measure different things:
- **Gain**: total improvement in the training objective attributable to splits using this feature, summed across all trees. Reflects how *structurally useful* a feature was for building the model.
- **SHAP**: average magnitude of a feature's contribution to individual prediction outputs. Reflects how much a feature *actually moves predictions* for real examples.

A feature can be secondary by gain but still meaningfully shift predictions — which is what happened here. This is reported as a genuine, nuanced finding rather than forced into a simple "worked" / "didn't work" conclusion.

---

## 4. Threshold selection: cost-based, not F1-based

### 4.1 Why not just optimize F1
F1 treats a false negative and false positive as equally costly, which isn't true in fraud detection — missing a $3,000 fraud is worse than one extra false alarm.

### 4.2 Cost model
```
Total cost = Σ(Amount of missed fraud) + (# false positives × investigation_cost)
```
- `investigation_cost` is set to a flat $5 — a simplifying assumption representing the operational cost of reviewing a flagged transaction (analyst time, customer friction). A real institution would have actual data for this figure; here it's a reasonable placeholder, documented explicitly as an assumption.
- The threshold is swept from 0.01 to 0.99 and the one minimizing total cost is selected.

### 4.3 Result
- Default threshold (0.5): estimated loss $2,730.05 on the test set
- Cost-optimal threshold (0.75): estimated loss $2,676.05
- Savings: $54.00

The dollar figure is modest because the test set is small (56,962 transactions, 75 fraud cases) — the value of this section is the **methodology** (optimizing against a real cost function instead of an abstract metric), not the absolute dollar amount, which would scale with a larger deployment.

---

## 5. Calibration

`calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')` — bins the *predicted probability axis* evenly (not by equal sample count), which matters because with only 75 positive test examples, quantile-based binning collapses most bins into the near-zero-probability region and produces an uninformative plot.

**Read of the result**: well-calibrated in the low-probability range (which contains the vast majority of transactions and the most statistically reliable data). Noisier in the high-probability range (0.65–1.0) — attributable to small sample size in those bins rather than a genuine calibration failure. This is stated explicitly rather than smoothed over.

To see exactly how few examples back the high-confidence bins:
```python
bin_counts = np.histogram(y_proba, bins=10, range=(0, 1))[0]
print("Test examples per probability bin:", bin_counts)
```

---

## 6. Explainability

`shap.TreeExplainer` computes exact SHAP values for XGBoost (no approximation needed, unlike model-agnostic SHAP methods). Two outputs:
- **Global summary** (`shap_summary.png`): every test example's SHAP value per feature, showing both magnitude and direction of impact across the dataset.
- **Single-example explanation** (`shap_example_explanation.png`): a force plot for one specific fraud case, breaking down exactly which features pushed the prediction toward "fraud" and by how much. This is the most demoable artifact in the project — it turns "the model predicts fraud" into "the model predicts fraud *because of these specific factors*."

---

## 7. Known limitations (stated deliberately, not hidden)

- All features except `Time` and `Amount` are anonymized PCA components — no domain-informed feature engineering was possible (e.g. merchant category, geographic distance between transactions).
- Two days of data — no ability to model longer-term drift, seasonality, or evolving fraud patterns.
- The $5 investigation cost is an assumption, not sourced from a real institution.
- Calibration conclusions above ~0.7 predicted probability carry high uncertainty due to small positive-class sample size (75 examples in test).

---

## 8. File reference

| File | Purpose |
|---|---|
| `fraud_detection_hybrid_pipeline.py` | Full pipeline: load → split → train → evaluate → threshold → calibrate → explain |
| `outputs/cost_threshold_curve.png` | Cost vs. threshold sweep |
| `outputs/confusion_matrix.png` | At the cost-optimal threshold |
| `outputs/calibration_curve.png` | Predicted probability vs. actual fraud rate |
| `outputs/shap_summary.png` | Global feature importance |
| `outputs/shap_example_explanation.png` | Single-prediction breakdown |
| `fraud_detection_hybrid_model.json` | Trained XGBoost model, reloadable via `xgb.XGBClassifier().load_model(...)` |
