import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- データ読み込み
df = pd.read_csv("features.csv")

# --- 特徴量とラベルを分離
feature_cols = [
    'length', 'FKGL', 'FRE', 'avg_clauses',#'ppl'
]
#    'length', 'avg_sentence_length', 'FKGL', 'FRE',
#    'avg_clauses', 'cwe', 'ppl'
X = df[feature_cols].fillna(0)  # 欠損値は0で補完

# ラベルの数値化（abstract=0, summary=1）
y = df['type'].map({'abstract': 0, 'summary': 1})

# --- 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 学習・テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# --- ロジスティック回帰モデルの訓練
model = LogisticRegression()
model.fit(X_train, y_train)

# --- 評価
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['abstract', 'summary']))

# --- 混同行列の可視化
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['abstract', 'summary'], yticklabels=['abstract', 'summary'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# --- モデル係数の確認
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_[0]
})
print("\n=== Feature Coefficients ===")
print(coef_df.sort_values(by='Coefficient', key=abs, ascending=False))
