# ===============================
# AI Banking Fraud Detection
# ===============================

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1 Load Dataset
# ===============================

data = pd.read_csv("bank_fraud_data.csv")

print("Dataset Preview")
print(data.head())

# ===============================
# 2 Feature Engineering
# ===============================

# Convert time to hour
data["Transaction_Hour"] = data["Transaction_Time"].apply(lambda x: int(x.split(":")[0]))

# Night transaction feature
data["Night_Transaction"] = data["Transaction_Hour"].apply(lambda x: 1 if x < 4 else 0)

# Drop unnecessary column
data = data.drop(["Transaction_Time","Transaction_ID"], axis=1)

# ===============================
# 3 Encode Categorical Features
# ===============================

le = LabelEncoder()

data["Customer_ID"] = le.fit_transform(data["Customer_ID"])
data["Merchant_Category"] = le.fit_transform(data["Merchant_Category"])
data["Transaction_Type"] = le.fit_transform(data["Transaction_Type"])
data["Location"] = le.fit_transform(data["Location"])
data["Device_Type"] = le.fit_transform(data["Device_Type"])

# ===============================
# 4 Split Features and Target
# ===============================

X = data.drop("Fraud_Label", axis=1)
y = data["Fraud_Label"]

# ===============================
# 5 Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6 Handle Imbalanced Data
# ===============================

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ===============================
# 7 Feature Scaling
# ===============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 8 Train AI Model (XGBoost)
# ===============================

model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ===============================
# 9 Model Prediction
# ===============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ===============================
# 10 Model Evaluation
# ===============================

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# 11 Feature Importance
# ===============================

importance = model.feature_importances_

features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()

# ===============================
# 12 Fraud Prediction Example
# ===============================

sample_transaction = X_test[0].reshape(1,-1)

prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("⚠ Fraudulent Transaction Detected")
else:
    print("✓ Legitimate Transaction")