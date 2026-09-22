import pandas as pd
import numpy as np
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

# 1. Dataset Load & Committed Offline Fallback Export
try:
    df = sns.load_dataset('titanic')
except Exception:
    df = pd.read_csv("titanic.csv")

# Save offline fallback
df.to_csv("titanic.csv", index=False)

# 2. Missing-Value Handling per Threshold Rules
# <5% missing -> drop rows; 5%-30% -> impute; >30% -> drop column
missing_pct = (df.isnull().sum() / len(df)) * 100
print("Missing percentages:\n", missing_pct[missing_pct > 0])

# deck is ~77% missing -> drop column
df_clean = df.drop(columns=['deck'])
# embarked and embark_town are <5% missing (0.22%) -> drop missing rows
df_clean = df_clean.dropna(subset=['embarked', 'embark_town'])

# 3. IQR-based Outlier Analysis for age and fare
for col in ['age', 'fare']:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df_clean[(df_clean[col] < q1 - 1.5 * iqr) | (df_clean[col] > q3 + 1.5 * iqr)]
    print(f"{col} Outliers (IQR rule): {len(outliers)}")

fare_mean = df_clean['fare'].mean()
fare_median = df_clean['fare'].median()
fare_mode = df_clean['fare'].mode()[0]
print(f"Fare Distribution: Mean={fare_mean:.2f}, Median={fare_median:.2f}, Mode={fare_mode:.2f} -> Right-Skewed (Mean > Median > Mode)")

# 4. Bivariate Survival Breakdown
print("\nSurvival rate by Sex:\n", df_clean.groupby('sex')['survived'].mean())
print("\nSurvival rate by Pclass:\n", df_clean.groupby('pclass')['survived'].mean())
print("\nSurvival rate by Sex and Pclass:\n", df_clean.groupby(['sex', 'pclass'])['survived'].mean())

# Correlation Matrix on the 6 requested numeric columns (adult_male and alone excluded)
corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_mat = df_clean[corr_cols].corr()
print("\n6x6 Correlation Matrix:\n", corr_mat)

# 5. Predictive Modeling Pipeline (Fit on train only)
X = df_clean[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df_clean['survived']

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = ['age', 'fare', 'sibsp', 'parch']
cat_features = ['sex', 'embarked', 'pclass']

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
for name, clf in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    })

print("\nModel Evaluation Summary:\n", pd.DataFrame(results))

# 6. Imbalance Handling Comparison (Train split only)
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# (a) Baseline
rf_base = RandomForestClassifier(random_state=42).fit(X_train_trans, y_train)
# (b) Balanced Class Weight
rf_bal = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X_train_trans, y_train)
# (c) SMOTE oversampling
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train_trans, y_train)
rf_smote = RandomForestClassifier(random_state=42).fit(X_train_sm, y_train_sm)

print("\nImbalance Handling (F1 comparison):")
print("Baseline:", f1_score(y_test, rf_base.predict(X_test_trans)))
print("Class Weight Balanced:", f1_score(y_test, rf_bal.predict(X_test_trans)))
print("SMOTE:", f1_score(y_test, rf_smote.predict(X_test_trans)))

# 7. Hyperparameter Tuning with OOB Score
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__max_features': ['sqrt', 'log2']
}
rf_oob_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(oob_score=True, random_state=42))
])
grid_search = GridSearchCV(rf_oob_pipe, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("OOB Score:", best_model.named_steps['classifier'].oob_score_)

# 8. Regression Side-Task: Predict Fare
y_reg = df_clean['fare']
X_reg = df_clean[['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_cat = ['sex', 'embarked', 'pclass']
reg_num = ['age', 'sibsp', 'parch']
reg_preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), reg_num),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), reg_cat)
])

reg_pipeline = Pipeline([('prep', reg_preprocessor), ('regressor', LinearRegression())])
reg_pipeline.fit(X_train_r, y_train_r)
y_reg_pred = reg_pipeline.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_test_r, y_reg_pred))
r2 = r2_score(y_test_r, y_reg_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test_r) - 1) / (len(y_test_r) - X_test_r.shape[1] - 1)
print(f"\nFare Regression Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}, Adj-R2={adj_r2:.2f}")

# 9. Save and Reload Complete Pipeline
joblib.dump(best_model, "best_titanic_pipeline.joblib")
loaded_pipeline = joblib.load("best_titanic_pipeline.joblib")
print("Verification on raw input sample:", loaded_pipeline.predict(X_test.head(1)))
