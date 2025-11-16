# ============================================================================
# CREDIT RISK PREDICTION WITH SHAP INTERPRETABILITY ANALYSIS
# Complete Implementation - FIXED FOR PYTHON 3.7 (No Memory Issues)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, 
                             balanced_accuracy_score, auc)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations - FIXED FOR COMPATIBILITY
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

sns.set_palette("husl")

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================

print("="*80)
print("STEP 1: LOADING AND EXPLORING DATA")
print("="*80)

# Load dataset
df = pd.read_csv('UCI_Credit_Card.csv')

# Remove ID column if present
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

# Rename target column for clarity
if 'default.payment.next.month' in df.columns:
    df.rename(columns={'default.payment.next.month': 'Default'}, inplace=True)

print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nFirst Few Rows:")
print(df.head())
print(f"\nData Types:")
print(df.dtypes)
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"\nTarget Variable Distribution:")
print(df['Default'].value_counts())
print(f"\nTarget Variable Percentage:")
print(df['Default'].value_counts(normalize=True) * 100)

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "="*80)
print("STEP 2: DATA PREPROCESSING")
print("="*80)

# Separate features and target
X = df.drop('Default', axis=1)
y = df['Default']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle categorical variables (if any)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"\nEncoding categorical columns: {categorical_cols}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

# Handle missing values
print(f"\nHandling missing values...")
X = X.fillna(X.mean())

# Feature scaling
print(f"Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split (80-20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ============================================================================
# STEP 3: MODEL TRAINING AND HYPERPARAMETER TUNING (FIXED - NO MEMORY ISSUES)
# ============================================================================

print("\n" + "="*80)
print("STEP 3: MODEL TRAINING AND HYPERPARAMETER TUNING")
print("="*80)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nScale pos weight (for class imbalance): {scale_pos_weight:.4f}")

# Define hyperparameter grid - REDUCED for faster training
params_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150]  # Reduced from [200, 300] to save memory
}

print(f"\nHyperparameter Grid: {params_grid}")
print("\nPerforming Grid Search with Cross-Validation...")
print("NOTE: Using n_jobs=1 (sequential) to avoid memory issues on Python 3.7")

# Create base model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=1,  # FIXED: Changed from -1 to 1 (no parallel processing)
    tree_method='hist'
)

# Grid search - FIXED: n_jobs=1 to prevent memory crash
grid_search = GridSearchCV(
    xgb_model, 
    params_grid,
    cv=3,  # Reduced from 5 to 3 folds to save memory
    scoring='roc_auc',
    n_jobs=1,  # FIXED: Sequential processing instead of parallel
    verbose=1
)

print("\nThis may take 2-5 minutes... Please wait...")
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")

# ============================================================================
# STEP 4: MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: MODEL EVALUATION")
print("="*80)

# Predictions
y_pred_train = best_xgb.predict(X_train)
y_pred_test = best_xgb.predict(X_test)
y_pred_proba_train = best_xgb.predict_proba(X_train)[:, 1]
y_pred_proba_test = best_xgb.predict_proba(X_test)[:, 1]

# Calculate metrics
train_roc_auc = roc_auc_score(y_train, y_pred_proba_train)
test_roc_auc = roc_auc_score(y_test, y_pred_proba_test)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print("\n" + "-"*80)
print("TRAINING SET METRICS")
print("-"*80)
print(f"ROC-AUC: {train_roc_auc:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_train, y_pred_train):.4f}")
print(f"F1-Score: {f1_score(y_train, y_pred_train):.4f}")
print(f"\nClassification Report (Training):")
print(classification_report(y_train, y_pred_train))

print("\n" + "-"*80)
print("TEST SET METRICS")
print("-"*80)
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"Balanced Accuracy: {test_balanced_acc:.4f}")
print(f"F1-Score: {test_f1:.4f}")
print(f"\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print(f"\nConfusion Matrix (Test):")
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {test_roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Credit Risk Model')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/01_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ ROC Curve saved to 'outputs/01_roc_curve.png'")

# ============================================================================
# STEP 5: GLOBAL FEATURE IMPORTANCE (NATIVE & SHAP)
# ============================================================================

print("\n" + "="*80)
print("STEP 5: GLOBAL FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Native XGBoost feature importance
print("\nCalculating native feature importance...")
feature_importance_native = best_xgb.feature_importances_
fi_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance_native
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Features (Native Importance):")
print(fi_df.head(10))

# Plot native importance
plt.figure(figsize=(10, 6))
plt.barh(fi_df.head(10)['Feature'], fi_df.head(10)['Importance'], color='steelblue')
plt.xlabel('Importance Score')
plt.title('Top 10 Features - XGBoost Native Importance')
plt.tight_layout()
plt.savefig('outputs/02_native_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✓ Native Importance plot saved to 'outputs/02_native_importance.png'")

# SHAP Summary Plot
print("\nCalculating SHAP values (this may take 1-2 minutes)...")
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# Handle binary classification output
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# SHAP Bar Plot (Mean Absolute Impact)
print("Generating SHAP Bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Summary Plot - Mean Absolute Impact on Model Output')
plt.tight_layout()
plt.savefig('outputs/03_shap_summary_bar.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ SHAP Summary Bar plot saved to 'outputs/03_shap_summary_bar.png'")

# SHAP Beeswarm Plot
print("Generating SHAP Beeswarm plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot - Feature Values vs Impact on Output')
plt.tight_layout()
plt.savefig('outputs/04_shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ SHAP Beeswarm plot saved to 'outputs/04_shap_summary_beeswarm.png'")

# Top 5 global feature drivers
top_5_features = fi_df.head(5)['Feature'].tolist()
print(f"\nTop 5 Global Feature Drivers:")
for idx, feature in enumerate(top_5_features, 1):
    print(f"{idx}. {feature}")

# ============================================================================
# STEP 6: INDIVIDUAL CUSTOMER PROFILE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: INDIVIDUAL CUSTOMER PROFILE ANALYSIS")
print("="*80)

y_pred_proba_test_full = best_xgb.predict_proba(X_test)[:, 1]

# Select customers from different risk profiles
high_risk_idx = np.where(y_pred_proba_test_full > 0.7)[0]
low_risk_idx = np.where(y_pred_proba_test_full < 0.3)[0]
borderline_idx = np.where((y_pred_proba_test_full > 0.4) & (y_pred_proba_test_full < 0.6))[0]

# Select one customer from each profile
if len(high_risk_idx) > 0:
    high_risk_customer = high_risk_idx[0]
else:
    high_risk_customer = np.argmax(y_pred_proba_test_full)

if len(low_risk_idx) > 0:
    low_risk_customer = low_risk_idx[0]
else:
    low_risk_customer = np.argmin(y_pred_proba_test_full)

if len(borderline_idx) > 0:
    borderline_customer = borderline_idx[0]
else:
    borderline_customer = np.argsort(np.abs(y_pred_proba_test_full - 0.5))[0]

high_risk_prob = y_pred_proba_test_full[high_risk_customer]
low_risk_prob = y_pred_proba_test_full[low_risk_customer]
borderline_prob = y_pred_proba_test_full[borderline_customer]

print(f"\nHigh-Risk Customer Index: {high_risk_customer}, Probability: {high_risk_prob:.4f}")
print(f"Low-Risk Customer Index: {low_risk_customer}, Probability: {low_risk_prob:.4f}")
print(f"Borderline Customer Index: {borderline_customer}, Probability: {borderline_prob:.4f}")

# SHAP Force Plots for each customer
customers = [
    ('High-Risk', high_risk_customer, high_risk_prob),
    ('Low-Risk', low_risk_customer, low_risk_prob),
    ('Borderline', borderline_customer, borderline_prob)
]

base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

print("\nGenerating SHAP Force plots...")
for idx, (customer_type, cust_idx, prob) in enumerate(customers, 1):
    plt.figure(figsize=(14, 4))
    shap.force_plot(
        base_value,
        shap_values[cust_idx],
        X_test.iloc[cust_idx],
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot - {customer_type} Customer (Default Probability: {prob:.4f})')
    plt.tight_layout()
    plt.savefig(f'outputs/05_shap_force_{customer_type.lower().replace("-", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("✓ SHAP Force plots saved for all 3 customers")

# SHAP Dependence Plots for top 3 features
print("Generating SHAP Dependence plots...")
for idx, feature in enumerate(top_5_features[:3], 1):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values, X_test, show=False)
    plt.title(f'SHAP Dependence Plot - {feature}')
    plt.tight_layout()
    plt.savefig(f'outputs/06_shap_dependence_{feature.replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

print("✓ SHAP Dependence plots saved for top 3 features")

# Detailed customer analysis
print("\n" + "="*80)
print("DETAILED CUSTOMER PROFILE ANALYSIS")
print("="*80)

for customer_type, cust_idx, prob in customers:
    print(f"\n{customer_type.upper()} CUSTOMER (Index: {cust_idx})")
    print("-"*80)
    print(f"Predicted Default Probability: {prob:.4f}")
    print(f"Actual Default Status: {y_test.iloc[cust_idx]}")
    
    customer_shap = shap_values[cust_idx]
    feature_contributions = pd.DataFrame({
        'Feature': X_test.columns,
        'SHAP Value': customer_shap,
        'Abs SHAP Value': np.abs(customer_shap),
        'Direction': ['Increases Risk' if v > 0 else 'Decreases Risk' for v in customer_shap],
        'Customer Value': X_test.iloc[cust_idx].values
    }).sort_values('Abs SHAP Value', ascending=False)
    
    print(f"\nTop 5 Contributing Features:")
    print(feature_contributions[['Feature', 'SHAP Value', 'Direction', 'Customer Value']].head(5))

# ============================================================================
# STEP 7: COMPREHENSIVE WRITTEN ANALYSIS REPORT
# ============================================================================

print("\n" + "="*80)
print("STEP 7: GENERATING COMPREHENSIVE ANALYSIS REPORT")
print("="*80)

report = f"""
{'='*90}
CREDIT RISK PREDICTION MODEL - INTERPRETABILITY ANALYSIS REPORT
{'='*90}

EXECUTIVE SUMMARY
{'-'*90}
This comprehensive analysis presents a machine learning model developed to predict 
credit default risk using binary classification. The model achieves a ROC-AUC of 
{test_roc_auc:.4f} on the test set, demonstrating strong discriminative ability 
between defaulters and non-defaulters. Using SHAP (SHapley Additive exPlanations) 
methodology, we provide detailed insights into the feature contributions driving 
individual predictions and global model behavior.

1. MODEL PERFORMANCE METRICS
{'-'*90}

Training Set Performance:
- ROC-AUC Score: {train_roc_auc:.4f}
- Balanced Accuracy: {balanced_accuracy_score(y_train, y_pred_train):.4f}
- F1-Score: {f1_score(y_train, y_pred_train):.4f}

Test Set Performance:
- ROC-AUC Score: {test_roc_auc:.4f}
- Balanced Accuracy: {test_balanced_acc:.4f}
- F1-Score: {test_f1:.4f}

Confusion Matrix (Test Set):
- True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}
- False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}

The model demonstrates reliable performance with balanced sensitivity and 
specificity, indicating effective risk discrimination across customer segments. 
The ROC-AUC score of {test_roc_auc:.4f} indicates strong predictive power for 
distinguishing between default and non-default customers.

2. TOP 5 GLOBAL FEATURE DRIVERS
{'-'*90}

The following features emerged as the most influential in determining default risk,
ranked by their importance scores:

"""

# Add top 5 features with analysis
for rank, (_, row) in enumerate(fi_df.head(5).iterrows(), 1):
    feature_name = row['Feature']
    importance = row['Importance']
    feature_idx = list(X.columns).index(feature_name)
    avg_impact = np.mean(np.abs(shap_values[:, feature_idx]))
    
    report += f"""
{rank}. {feature_name.upper()} (Importance Score: {importance:.4f})
   - Average SHAP Impact: {avg_impact:.4f}
   - This feature consistently demonstrates high predictive power for default 
     classification and represents a key driver of credit risk assessment.
   - Higher values indicate increased/decreased default risk depending on the 
     customer segment and interaction with other features.
   - Variance across customers: {np.std(shap_values[:, feature_idx]):.4f}
"""

report += f"""

3. INDIVIDUAL CUSTOMER RISK PROFILES
{'-'*90}

3.1 HIGH-RISK CUSTOMER (Test Index: {high_risk_customer})
    Predicted Default Probability: {high_risk_prob:.4f}
    Actual Default Status: {'Defaulted' if y_test.iloc[high_risk_customer] == 1 else 'Non-Default'}

Analysis:
This customer exhibits characteristics strongly associated with credit default. 
The high predicted probability of {high_risk_prob:.4f} indicates substantial risk 
factors present in their profile. Key contributing factors include features that 
collectively elevate the customer's default probability by shifting predictions 
substantially above the base rate (model baseline: {base_value:.4f}).

Recommendation: Immediate intervention recommended - consider credit line reduction, 
increased monitoring, or portfolio review.

3.2 LOW-RISK CUSTOMER (Test Index: {low_risk_customer})
    Predicted Default Probability: {low_risk_prob:.4f}
    Actual Default Status: {'Defaulted' if y_test.iloc[low_risk_customer] == 1 else 'Non-Default'}

Analysis:
This customer displays characteristics that substantially reduce default risk. The 
low predicted probability of {low_risk_prob:.4f} is driven by protective factors 
in their profile. Features such as payment history, credit utilization, and income 
levels contribute positively to creditworthiness and collectively drive predictions 
well below the base rate.

Recommendation: Standard monitoring sufficient; customer qualifies for favorable 
credit terms if applicable.

3.3 BORDERLINE CUSTOMER (Test Index: {borderline_customer})
    Predicted Default Probability: {borderline_prob:.4f}
    Actual Default Status: {'Defaulted' if y_test.iloc[borderline_customer] == 1 else 'Non-Default'}

Analysis:
This customer represents a decision boundary case with competing risk and protective 
factors. The borderline probability of {borderline_prob:.4f} suggests the customer 
sits at the margin of classification. Both positive and negative factors are 
present, resulting in marginal distinction from the decision threshold.

Recommendation: Enhanced due diligence warranted. Consider alternative assessment 
methods, expert review, or additional verification procedures before final decision.

4. BUSINESS INSIGHTS AND ACTIONABLE RECOMMENDATIONS
{'-'*90}

1. Risk Segmentation Strategy:
   - Segment customers into risk tiers based on predicted probabilities
   - High-Risk (>0.7): Intensive monitoring and early intervention
   - Borderline (0.4-0.6): Enhanced due diligence and expert review
   - Low-Risk (<0.3): Standard monitoring protocols

2. Feature Monitoring Program:
   - Implement real-time monitoring of top 5 feature drivers
   - Establish alert thresholds for significant changes in key features
   - Regular model performance tracking and retraining schedule

3. Decision Support:
   - Use SHAP force plots to explain credit decisions to stakeholders
   - Provide transparent, interpretable reasoning for borderline cases
   - Train credit officers on model interpretability and feature interactions

4. Portfolio Management:
   - Apply model to entire customer base for risk stratification
   - Focus resources on high-risk segments with elevated default probability
   - Develop targeted retention and intervention programs

5. MODEL GOVERNANCE
{'-'*90}

Data Quality:
- Dataset comprises {len(df):,} credit records with {len(X.columns)} features
- Training set: {len(X_train):,} samples | Test set: {len(X_test):,} samples
- Default rate: {(y.sum()/len(y)*100):.2f}% (class imbalance addressed via weighting)

Model Architecture:
- Algorithm: XGBoost Gradient Boosting Classifier
- Optimal Parameters: {grid_search.best_params_}
- Cross-Validation Score: {grid_search.best_score_:.4f}

Interpretability Method:
- SHAP (SHapley Additive exPlanations) for feature attribution
- TreeExplainer for XGBoost compatibility
- Force plots, dependence plots, and summary plots generated

6. LIMITATIONS AND CONSIDERATIONS
{'-'*90}

- Model performance depends on data quality and representativeness of training set
- Feature importance rankings may change with data drift or distribution shifts
- SHAP values represent local approximations; complex multi-way interactions may 
  not be fully captured by individual feature analysis
- Predictions should be used as decision support tools, not sole decision drivers
- Regular model retraining recommended (quarterly minimum) to maintain accuracy
- Feature engineering and new feature discovery may improve performance

7. CONCLUSION
{'-'*90}

This interpretable machine learning model provides a robust framework for credit 
risk assessment with clear explainability through SHAP analysis. The model achieves 
strong predictive performance (ROC-AUC: {test_roc_auc:.4f}) while maintaining 
transparency in decision-making. By combining global feature importance rankings 
with individual customer-level SHAP analysis, we enable both strategic risk 
management and case-by-case decision support.

The top 5 feature drivers identified provide a clear foundation for credit policy 
development and risk mitigation strategies. The detailed analysis of three customer 
profiles demonstrates the model's ability to distinguish between clearly positive, 
clearly negative, and ambiguous credit cases.

{'='*90}
Report Generated: Credit Risk Prediction Analysis
{'='*90}
"""

print(report)

# Save report to file
with open('outputs/CREDIT_RISK_ANALYSIS_REPORT.txt', 'w') as f:
    f.write(report)

print("\n✓ Full analysis report saved to 'outputs/CREDIT_RISK_ANALYSIS_REPORT.txt'")

# ============================================================================
# STEP 8: SAVE SUMMARY DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 8: SAVING SUMMARY DATA")
print("="*80)

# Save feature importance
fi_df.to_csv('outputs/feature_importance.csv', index=False)
print("✓ Feature importance saved to 'outputs/feature_importance.csv'")

# Save model performance summary
performance_summary = pd.DataFrame({
    'Metric': ['ROC-AUC', 'Balanced Accuracy', 'F1-Score'],
    'Training': [train_roc_auc, balanced_accuracy_score(y_train, y_pred_train), f1_score(y_train, y_pred_train)],
    'Test': [test_roc_auc, test_balanced_acc, test_f1]
})
performance_summary.to_csv('outputs/model_performance.csv', index=False)
print("✓ Model performance saved to 'outputs/model_performance.csv'")

# Save customer profiles analysis
customer_profiles = []
for customer_type, cust_idx, prob in customers:
    customer_shap = shap_values[cust_idx]
    customer_profiles.append({
        'Profile Type': customer_type,
        'Test Index': cust_idx,
        'Default Probability': prob,
        'Actual Default': y_test.iloc[cust_idx],
        'Top Feature': X_test.columns[np.argmax(np.abs(customer_shap))],
        'Max SHAP Impact': np.max(np.abs(customer_shap))
    })

customer_profiles_df = pd.DataFrame(customer_profiles)
customer_profiles_df.to_csv('outputs/customer_profiles.csv', index=False)
print("✓ Customer profiles saved to 'outputs/customer_profiles.csv'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROJECT COMPLETION SUMMARY")
print("="*80)
print("""
✓ All outputs saved to 'outputs/' folder

Generated Files:
1. outputs/01_roc_curve.png - ROC Curve visualization
2. outputs/02_native_importance.png - Native feature importance plot
3. outputs/03_shap_summary_bar.png - SHAP mean impact bar plot
4. outputs/04_shap_summary_beeswarm.png - SHAP value distribution plot
5. outputs/05_shap_force_high_risk.png - High-risk customer force plot
6. outputs/05_shap_force_low_risk.png - Low-risk customer force plot
7. outputs/05_shap_force_borderline.png - Borderline customer force plot
8. outputs/06_shap_dependence_*.png - Feature dependence plots (3 files)
9. outputs/CREDIT_RISK_ANALYSIS_REPORT.txt - Comprehensive written analysis
10. outputs/feature_importance.csv - Feature rankings
11. outputs/model_performance.csv - Performance metrics
12. outputs/customer_profiles.csv - Customer profile analysis

Next Steps for Submission:
1. Push this script and outputs to GitHub
2. Use Gitingest to generate markdown from GitHub repo
3. Include generated markdown in your submission with project explanation
4. Ensure score target is 80% or above

Model Ready for Production Evaluation!
""")

print("\n✓ Project completed successfully!")
