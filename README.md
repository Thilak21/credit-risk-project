# Credit Risk Prediction with SHAP Analysis

## Project Overview

This project implements an interpretable machine learning solution for credit default risk prediction. The system combines XGBoost gradient boosting with SHAP (SHapley Additive exPlanations) methodology to deliver both high predictive accuracy and complete model transparency.

The solution analyzes customer credit patterns to identify default risk while providing explicit explanations for each prediction. This enables credit officers to make informed, defensible decisions backed by clear reasoning and regulatory compliance documentation.


**Objective**: Predict customer credit default probability while maintaining complete interpretability of model decisions.


## Model Performance

### Test Set Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| ROC-AUC | 0.7760 | 77.6% discriminative ability between defaulters and non-defaulters |
| Balanced Accuracy | 0.7079 | 70.8% average of sensitivity and specificity |
| F1-Score | 0.5321 | Balanced precision-recall performance on minority class |
| Overall Accuracy | 0.7600 | 76% correct classifications across all customers |

### Training Set Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8165 |
| Balanced Accuracy | 0.7356 |
| F1-Score | 0.5690 |

### Confusion Matrix (Test Set)

```
                    Predicted Non-Default    Predicted Default
Actual Non-Default           3,771                  902
Actual Default                519                  808
```

The model correctly identifies 61% of actual defaulters while maintaining 81% specificity for non-defaulters.

---

## Dataset

### UCI Credit Card Default Dataset

- Total Records: 30,000 customers
- Total Features: 24 (23 after preprocessing)
- Default Rate: 22.12%
- Non-Default Rate: 77.88%
- Missing Values: None
- Feature Type: All numeric

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| Account Information | LIMIT_BAL, AGE | Credit limit and customer age |
| Demographics | SEX, EDUCATION, MARRIAGE | Customer demographic attributes |
| Payment Status | PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 | Monthly payment status history (6 months) |
| Bill Amounts | BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6 | Monthly statement amounts |
| Payment Amounts | PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6 | Monthly payment amounts |

---

## Top Feature Drivers

### Global Feature Importance Rankings

| Rank | Feature | Importance Score | SHAP Impact | Interpretation |
|------|---------|------------------|-------------|-----------------|
| 1 | PAY_0 | 47.33% | 0.546 | Most recent payment status dominates predictions |
| 2 | PAY_4 | 5.72% | 0.055 | Payment status from 4 months prior |
| 3 | PAY_2 | 5.65% | 0.084 | Payment status from 2 months prior |
| 4 | PAY_3 | 4.84% | 0.078 | Payment status from 3 months prior |
| 5 | PAY_AMT2 | 4.05% | 0.106 | Payment amount from 2 months prior |

### Key Insight

Payment history features constitute 90% of top feature drivers. This indicates that recent payment behavior is the strongest predictor of credit default risk. The model prioritizes payment status and patterns over demographic or account-level features.

---

## Customer Risk Profiles

### High-Risk Customer (Index: 5)

- Predicted Default Probability: 79.59%
- Actual Default Status: Non-Default
- Primary Risk Factors:
  - Recent payment issues (PAY_0 contribution: +0.398)
  - Payment problems 2 months ago (PAY_2 contribution: +0.367)
  - High bill amounts (BILL_AMT2 contribution: +0.188)

**Recommendation**: Immediate intervention required. Consider credit line reduction, increased monitoring frequency, and proactive customer contact.

### Low-Risk Customer (Index: 4)

- Predicted Default Probability: 12.34%
- Actual Default Status: Non-Default
- Primary Protective Factors:
  - Excellent recent payment record (PAY_0 contribution: -0.507)
  - High credit limit relative to usage (LIMIT_BAL contribution: -0.352)
  - Recent payment made (PAY_AMT1 contribution: -0.213)

**Recommendation**: Standard monitoring protocols sufficient. Customer qualifies for premium credit terms and loyalty program consideration.

### Borderline Customer (Index: 3)

- Predicted Default Probability: 42.84%
- Actual Default Status: Defaulted
- Risk Factors: High bill amounts across multiple months (+0.328, +0.301)
- Protective Factors: Good payment status (-0.516), Adequate credit limit (-0.187)

**Recommendation**: Enhanced due diligence warranted. Manual expert review, income verification, and alternative assessment methods recommended.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Setup Instructions

**Step 1: Clone Repository**

```bash
git clone https://github.com/username/credit_risk_project.git
cd credit_risk_project
```

**Step 2: Download Dataset**

- Download UCI_Credit_Card.csv from Kaggle
- Link: https://www.kaggle.com/uciml/default-of-credit-card-clients
- Place in project root directory

**Step 3: Create Output Directory**

```bash
mkdir outputs
```

**Step 4: Install Dependencies**

```bash
pip install -r requirements.txt
```

This installs all required packages:
- pandas, numpy (data manipulation and numerical operations)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning preprocessing and metrics)
- xgboost (gradient boosting classifier)
- shap (model interpretability analysis)

### Verify Installation

```bash
python -c "import xgboost; import shap; print('Installation successful')"
```

---

## Running the Project

### Execute Full Pipeline

```bash
python main.py
```

### Expected Output

The script executes 8 sequential steps:

1. Load and explore dataset
2. Preprocess and scale features
3. Train model with hyperparameter tuning
4. Evaluate model performance
5. Calculate global feature importance
6. Analyze individual customer risk profiles
7. Generate comprehensive analysis report
8. Save all results to outputs folder

Expected runtime: 2-3 minutes

### Output Files

Upon completion, the outputs folder contains:

**Visualizations (8 PNG files)**:
- 01_roc_curve.png: Model performance ROC curve
- 02_native_importance.png: Feature importance rankings
- 03_shap_summary_bar.png: Mean absolute SHAP impact (global)
- 04_shap_summary_beeswarm.png: SHAP value distributions
- 05_shap_force_high_risk.png: High-risk customer explanation
- 05_shap_force_low_risk.png: Low-risk customer explanation
- 05_shap_force_borderline.png: Borderline customer explanation
- 06_shap_dependence_*.png: Feature dependence plots (3 files)

**Reports and Data (4 files)**:
- CREDIT_RISK_ANALYSIS_REPORT.txt: Comprehensive written analysis
- feature_importance.csv: Ranked features with importance scores
- model_performance.csv: Train vs test performance metrics
- customer_profiles.csv: Individual customer profile analysis

---

## Technology Stack

### Languages and Frameworks

- Python 3.7.8 (compatible with 3.8+)
- XGBoost for gradient boosting
- scikit-learn for preprocessing and evaluation

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| xgboost | Latest | Gradient boosting classifier |
| scikit-learn | Latest | Data preprocessing and metrics |
| shap | Latest | Model interpretability via SHAP values |
| pandas | Latest | Data manipulation and analysis |
| numpy | Latest | Numerical computations |
| matplotlib | Latest | Plot generation |
| seaborn | Latest | Statistical data visualization |

---

## Model Architecture

### Algorithm: XGBoost Classifier

XGBoost was selected for this application based on:

- Superior performance on tabular/structured data
- Native feature importance calculation
- Direct compatibility with SHAP via TreeExplainer
- Built-in class imbalance handling
- Fast training and inference

### Optimal Hyperparameters

```
learning_rate:       0.05        (Step size per boosting iteration)
max_depth:           4           (Tree depth constraint)
n_estimators:        150         (Number of boosting rounds)
scale_pos_weight:    3.52        (Class imbalance weight)
random_state:        42          (Reproducibility seed)
cv_folds:            3           (Cross-validation folds)
train_test_ratio:    0.8/0.2     (80% train, 20% test)
```

### Cross-Validation Strategy

- Method: 3-fold stratified cross-validation
- CV ROC-AUC Score: 0.7807
- Test ROC-AUC Score: 0.7760
- Overfitting Gap: 0.0047 (negligible)

---

## SHAP Interpretability

### Methodology

SHAP (SHapley Additive exPlanations) applies game theory concepts to calculate each feature's contribution to individual predictions. The methodology enables both global and local model interpretability.

### SHAP Components Generated

**Global Explanations**:
- Bar Plot: Average feature impact magnitude across all customers
- Beeswarm Plot: Distribution of feature impacts and value relationships

**Local Explanations**:
- Force Plots: Individual customer prediction decomposition
- Dependence Plots: Feature value versus SHAP impact relationships

### Interpretation Guide

- Base Value: Model baseline prediction (0.0012 or 0.12%)
- Positive SHAP Values: Features increasing default probability
- Negative SHAP Values: Features decreasing default probability
- Magnitude: Larger absolute values indicate stronger influence

---

## Project Structure

```
credit_risk_project/
├── README.md                              (Project documentation)
├── SUBMISSION.md                          (Detailed analysis report)
├── requirements.txt                       (Package dependencies)
├── main.py                                (Complete implementation)
├── UCI_Credit_Card.csv                    (Dataset file)
│
└── outputs/                               (Generated results)
    ├── 01_roc_curve.png
    ├── 02_native_importance.png
    ├── 03_shap_summary_bar.png
    ├── 04_shap_summary_beeswarm.png
    ├── 05_shap_force_high_risk.png
    ├── 05_shap_force_low_risk.png
    ├── 05_shap_force_borderline.png
    ├── 06_shap_dependence_*.png (3 files)
    ├── CREDIT_RISK_ANALYSIS_REPORT.txt
    ├── feature_importance.csv
    ├── model_performance.csv
    └── customer_profiles.csv
```

---

## Reproducibility

### Deterministic Execution

All random seeds are fixed to ensure identical results across multiple executions:

- Train-test split randomization: seed=42
- Cross-validation fold creation: seed=42
- Model training initialization: seed=42

### Verified Reproducibility

```bash
# Run 1
python main.py
# ROC-AUC: 0.7760

# Run 2
python main.py
# ROC-AUC: 0.7760 (identical)
```

---

## Model Limitations

### Acknowledged Constraints

1. Historical Data: Model trained on historical patterns; cannot predict unprecedented events
2. Temporal Stability: Assumes payment patterns remain consistent (quarterly retraining recommended)
3. Feature Scope: Limited to behavioral data; alternative data sources not incorporated
4. Interaction Effects: SHAP analysis emphasizes main effects over complex multi-way interactions
5. False Positive Rate: 902 of 4,673 non-defaulters flagged (19% false positive rate)

### Mitigation Strategies

- Quarterly model retraining with fresh data
- Continuous performance monitoring and drift detection
- Manual expert review for borderline cases (0.4-0.6 probability)
- Fairness audits for demographic disparities
- Combination with alternative assessment methods

---

## Business Applications

### Credit Decision Support

For each customer application:
- Model generates default probability (0-100%)
- SHAP analysis identifies key decision drivers
- Force plots provide transparent reasoning
- Credit officers review and validate recommendations

### Risk Segmentation

- High-Risk (>70%): Intensive monitoring, reduced credit lines, proactive intervention
- Borderline (40-60%): Enhanced due diligence, manual review, verification required
- Low-Risk (<30%): Standard monitoring, favorable terms, loyalty programs

### Portfolio Management

- Score entire customer base by default probability
- Allocate risk management resources efficiently
- Track portfolio risk metrics
- Develop segment-specific intervention strategies

---

## Performance Monitoring

### Quarterly Assessment Tasks

1. Calculate model performance metrics on new data
2. Compare ROC-AUC, F1-Score, and other metrics to baseline
3. Monitor feature distributions for data drift
4. Conduct fairness audit for demographic bias
5. Verify calibration of predicted probabilities
6. Plan retraining if performance drops below 0.75 ROC-AUC

### Retraining Triggers

- ROC-AUC declines below 0.75
- Feature distributions change significantly
- Customer demographics shift materially
- Default rate changes beyond expected range
- New customer segments emerge in portfolio

---

## Dependencies

All required packages and versions are specified in requirements.txt:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Example 1: Basic Execution

```bash
cd credit_risk_project
pip install -r requirements.txt
python main.py
```

### Example 2: View Results

```bash
# Open ROC curve visualization
open outputs/01_roc_curve.png

# Read comprehensive analysis report
cat outputs/CREDIT_RISK_ANALYSIS_REPORT.txt

# Examine feature importance rankings
cat outputs/feature_importance.csv
```

### Example 3: Integrate into Workflow

```python
import xgboost as xgboost
import shap
import pickle

# Load trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

# Score new customer
prediction = model.predict_proba(new_customer_data)

# Generate SHAP explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(new_customer_data)
```

---

## Project Validation

### Code Quality Metrics

- PEP 8 Compliance: Maintained
- Code Documentation: Comprehensive comments
- Error Handling: Implemented throughout pipeline
- Reproducibility: All random seeds fixed

### Model Validation Results

- Cross-Validation ROC-AUC: 0.7807
- Test ROC-AUC: 0.7760
- Train-Test Gap: 0.0405 (no overfitting detected)
- Generalization Capability: Confirmed

---

## References

### Key Papers and Resources

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NIPS.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
3. UCI Machine Learning Repository: Default of Credit Card Clients Dataset

### External Links

- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- scikit-learn Documentation: https://scikit-learn.org/

---

## License

This project is provided for educational and commercial use.

---

## Conclusion

This project demonstrates production-ready interpretable machine learning for credit risk assessment. The model achieves 77.6% ROC-AUC performance while maintaining complete transparency through SHAP analysis. Implementation is suitable for pilot deployment with ongoing quarterly monitoring and retraining.

The combination of predictive accuracy and model explainability enables credit officers to make defensible, compliant decisions backed by clear, evidence-based reasoning.

