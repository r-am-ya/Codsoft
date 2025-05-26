import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib

# ===== LOAD DATASETS =====
try:
    train_df = pd.read_csv("fraudTrain.csv")
    test_df = pd.read_csv("fraudTest.csv")
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("❌ Please place 'fraudTrain.csv' and 'fraudTest.csv' in the project directory.")
    exit()

# Drop index columns if present
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# ===== FEATURE ENGINEERING =====
for df in [train_df, test_df]:
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['weekday'] = df['trans_date_trans_time'].dt.weekday
    df.drop(['trans_date_trans_time'], axis=1, inplace=True)

# ===== ENCODE CATEGORICAL COLUMNS =====
def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

train_df, _ = encode_categorical(train_df)
test_df, _ = encode_categorical(test_df)

# ===== MERGE AND CLEAN =====
df = pd.concat([train_df, test_df], ignore_index=True)

# Drop unnecessary columns (IDs, zip, etc.)
cols_to_drop = ['cc_num', 'street', 'zip', 'dob', 'trans_num', 'unix_time']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Scale 'amt'
scaler = StandardScaler()
df['amt'] = scaler.fit_transform(df[['amt']])

# ===== SPLIT FEATURES & TARGET =====
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("\n=== Class Distribution Before Resampling ===")
print(y_train.value_counts())

# ===== HANDLE CLASS IMBALANCE =====
resampling = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42))
])
X_train_res, y_train_res = resampling.fit_resample(X_train, y_train)

print("\n=== Class Distribution After Resampling ===")
print(pd.Series(y_train_res).value_counts())

# ===== LOGISTIC REGRESSION =====
print("\n=== Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nROC AUC Score: {:.4f}".format(roc_auc_score(y_test, y_prob_lr)))

# ===== RANDOM FOREST =====
print("\n=== Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nROC AUC Score: {:.4f}".format(roc_auc_score(y_test, y_prob_rf)))

# ===== FEATURE IMPORTANCE PLOT =====
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ===== PRECISION-RECALL CURVE =====
plt.figure(figsize=(10, 6))
precision, recall, _ = precision_recall_curve(y_test, y_prob_lr)
plt.plot(recall, precision, label=f'Logistic Regression (AUC = {auc(recall, precision):.2f})')

precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)
plt.plot(recall, precision, label=f'Random Forest (AUC = {auc(recall, precision):.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('precision_recall_curve.png')
plt.show()

# ===== SAVE BEST MODEL =====
joblib.dump(rf, 'fraud_detection_model.pkl')
print("\n✅ Model saved as 'fraud_detection_model.pkl'")


# ===== LOAD MODEL AND MAKE PREDICTION ON NEW SAMPLE =====
print("\n=== Predicting on New Sample Data ===")

# Load the saved model
loaded_model = joblib.load('fraud_detection_model.pkl')

# Prepare a new test sample — must match the features used in training
# You can extract this from your test data too
sample_data = X_test.iloc[[0]]  # Taking the first test row as an example

# Predict
sample_prediction = loaded_model.predict(sample_data)
print("🚨 Prediction for sample transaction:", "FRAUD" if sample_prediction[0] == 1 else "Legitimate")

# Optionally show prediction probability
sample_prob = loaded_model.predict_proba(sample_data)[0][1]
print(f"📊 Probability of Fraud: {sample_prob:.4f}")
