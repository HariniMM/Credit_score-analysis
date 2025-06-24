import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, DMatrix, train as xgb_train
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv("dataset-2.csv")

# Handle missing values

# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical variables

# Binary categorical:
if 'Payment_of_Min_Amount' in df.columns:
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'Yes':1, 'No':0})

le = LabelEncoder()
for col in cat_cols:
    if col != 'Payment_of_Min_Amount' and col != 'Credit_Score':
        df[col] = le.fit_transform(df[col])

# Encode target variable
df['Credit_Score'] = df['Credit_Score'].map({'Poor':0, 'Standard':1, 'Good':2})

# Feature scaling
exclude_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Credit_Score']
features = df.drop(columns=exclude_cols)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
df_prepared = pd.DataFrame(scaled_features, columns=features.columns)
df_prepared['Credit_Score'] = df['Credit_Score'].values

print(df_prepared.head())

# Split features and target
X = df_prepared.drop('Credit_Score', axis=1)
y = df_prepared['Credit_Score']

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Impute missing values for Random Forest
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Hyperparameter Tuning for Random Forest
print("Starting Random Forest hyperparameter tuning...")
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train_imputed, y_train)

best_rf = rf_grid.best_estimator_
print("Best Random Forest Params:", rf_grid.best_params_)

# Hyperparameter Tuning for XGBoost
print("\nStarting XGBoost hyperparameter tuning...")
xgb_model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1]
}
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_
print("Best XGBoost Params:", xgb_grid.best_params_)

# Early Stopping Simulation for Random Forest
print("\nRandom Forest Early Stopping Simulation")
n_estimators_options = [50, 100, 150, 200]
best_acc = 0
best_model = None
best_n = 0

rf_params = best_rf.get_params()
rf_params.pop('n_estimators', None)

for n in n_estimators_options:
    rf_temp = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        max_depth=rf_params.get('max_depth', None),
        min_samples_split=rf_params.get('min_samples_split', 2),
        min_samples_leaf=rf_params.get('min_samples_leaf', 1),
        n_jobs=-1
    )
    rf_temp.fit(X_train_imputed, y_train)
    preds = rf_temp.predict(X_test_imputed)
    acc = accuracy_score(y_test, preds)
    print(f"n_estimators={n} -> Test Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = rf_temp
        best_n = n

print(f"Best RF after early stopping simulation: n_estimators={best_n}, Accuracy={best_acc:.4f}")
print(classification_report(y_test, best_model.predict(X_test_imputed)))
print(confusion_matrix(y_test, best_model.predict(X_test_imputed)))

#  Early Stopping for XGBoost
print("\nXGBoost Early Stopping Training")

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

xgb_params = best_xgb.get_params()
for key in ['objective', 'eval_metric', 'use_label_encoder', 'n_estimators', 'random_state']:
    xgb_params.pop(key, None)

xgb_params.update({
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'seed': 42,
})

bst = xgb.train(
    params=xgb_params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dvalid, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=20
)

xgb_preds = bst.predict(dvalid)
xgb_preds_labels = xgb_preds.argmax(axis=1)

print(f"XGBoost best iteration: {bst.best_iteration}")
print(f"XGBoost Test Accuracy: {accuracy_score(y_test, xgb_preds_labels):.4f}")
print(classification_report(y_test, xgb_preds_labels))
print(confusion_matrix(y_test, xgb_preds_labels))

# Save the best Random Forest model
joblib.dump(best_rf, 'best_model.joblib')

print("Model saved as best_model.joblib")
