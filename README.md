# Stroke-Detection


# 🧠 Stroke Prediction Using Machine Learning

This project explores the development of a predictive machine learning model to estimate the risk of stroke using patient health and demographic data. Given the clinical importance of early detection, the focus is on maximizing **recall**, even at the expense of some false positives.

---

## 📌 Project Goal

To build an interpretable and balanced machine learning model that accurately identifies individuals at high risk of stroke, while handling class imbalance and providing insights through visual and statistical evaluation.

---

## 📊 Dataset Overview

- **Total observations**: 5,110
- **Target variable**: `stroke` (1 = stroke occurred, 0 = no stroke)
- **Class imbalance**: Only ~4.8% of individuals had a stroke

---

## ⚖️ Why SMOTE Was Used

The stroke class is significantly underrepresented (~5%). Using standard training without balancing leads to models that perform well on accuracy but **fail to detect stroke cases (low recall)**.

**SMOTE (Synthetic Minority Over-sampling Technique)**:
- Generates synthetic examples of the minority class in the training set.
- Prevents models from being biased toward the majority class.
- Increases model sensitivity to stroke cases.

🧠 **Result**: After applying SMOTE, the model could learn to detect stroke patterns better, improving **recall and F1-score** for the minority class.

---

## ⚙️ Preprocessing Steps

- Imputed missing BMI values using median.
- Capped and log-transformed outliers in `age`, `avg_glucose_level`, `bmi`.
- Encoded categorical variables using one-hot encoding.
- Standardized numerical features for linear models.
- Created two datasets: one for linear models, one for tree-based models.

---

## 🤖 Models Trained

- Logistic Regression ✅ (final model)
- Random Forest 🌲 (tested for comparison)
- XGBoost ⚡ (tested for comparison)

Initially, all models performed well on accuracy due to imbalance, but performed poorly in detecting stroke cases (low recall for class 1).

---

## 🎯 Threshold Tuning

At the default **threshold = 0.5**, the Logistic Regression model had:

| Metric     | Value     |
|------------|-----------|
| Precision (stroke=1) | ~0.14 |
| Recall (stroke=1)    | ~0.74 |
| F1 Score             | ~0.23 |
| Accuracy             | ~0.76 |

🔍 **Insight**: Very high recall but very low precision → too many false positives.

After tuning to **threshold = 0.8**, the model showed:

| Metric     | Value     |
|------------|-----------|
| Precision (stroke=1) | 0.23 |
| Recall (stroke=1)    | 0.44 |
| F1 Score             | 0.30 |
| Accuracy             | 0.90 |
| ROC AUC              | ~0.83 |

✅ **Better balance** — improved precision while still maintaining moderate recall.

---

## 📈 Graph Interpretations

### 🔷 Confusion Matrix
- Shows how many true stroke cases were correctly identified (TP), and how many were missed (FN).
- At threshold = 0.8, more positive cases are missed, but precision is higher.

### 🟠 ROC Curve
- Plots True Positive Rate vs False Positive Rate for all thresholds.
- **AUC ~0.83** → the model is good at ranking positive vs negative cases.

### 🟢 Precision-Recall Curve
- Visualizes the tradeoff between precision and recall.
- Useful when dealing with imbalanced datasets.
- Helps identify the **ideal threshold** that gives the best F1-score.

---

## ✅ Conclusion

- Logistic Regression with SMOTE and threshold tuning provides a strong balance for stroke detection.
- Prioritizing recall ensures fewer missed high-risk patients, which is essential in healthcare.
- AUC of 0.83 shows strong discriminatory ability.

---

## 🏥 Future Healthcare Applications

1. **Clinical Screening Tool** — EHR-integrated model to flag high-risk patients.
2. **Remote Monitoring** — Could power wearables/apps for continuous stroke risk updates.
3. **Population Health** — Applied to registries to identify high-risk groups for intervention.
4. **Decision Support** — Clinicians can use model outputs alongside other indicators.
5. **Explainability Tools** — Add SHAP or LIME to help clinicians trust and interpret predictions.

---

> Created with ❤️ using Python, pandas, scikit-learn, imbalanced-learn, and matplotlib.
