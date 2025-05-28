import os
import json
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # 非GUI環境でSHAP描画をサポート
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

# --- CSVファイルを読み込み、ラベルエンコードと特徴量抽出を行う
def load_and_prepare_data(input_csv):
    print(f"{datetime.datetime.now()}, load data from {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"{datetime.datetime.now()}, Failed to read CSV file: {e}")

    # ラベルを数値に変換
    label_map = { 'abstract': 0, 'summary-elife': 1, 'summary-plos': 2 }
    df['label'] = df['type'].map(label_map)

    if df['label'].isnull().any():
        raise ValueError(f"{datetime.datetime.now()}, Unknown types found in 'type' column.")

    # 特徴量カラムを抽出（doc_id, type, labelを除く）
    feature_columns = [col for col in df.columns if col not in ['doc_id', 'type', 'label']]

    print(f"--- Number of samples: {len(df)}")
    print(f"--- Extracted feature columns: {feature_columns}")
    print("--- Class distribution:")
    print(f"{df['type'].value_counts()}")

    return df, feature_columns

# 特徴量の標準化を行う関数
def preprocess_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return (
        pd.DataFrame(X_train_scaled, columns=X_train.columns),
        pd.DataFrame(X_test_scaled, columns=X_test.columns),
    )


def train_and_evaluate_rf(X_train, X_test, y_train, y_test, feature_combo, output_dir=None, combo_idx=None):
    X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    # === SHAP解析 ===
    if output_dir is not None and combo_idx is not None:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_scaled)

        shap_dir = os.path.join(output_dir, "shap_plots")
        os.makedirs(shap_dir, exist_ok=True)
        
        print(f"shap_values shape: {[sv.shape for sv in shap_values]}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")

        for class_idx, sv in enumerate(shap_values):
            plt.figure()
            shap.summary_plot(sv, X_test_scaled, show=False)  # X_test_scaled is now a DataFrame with correct column names
            plot_path = os.path.join(shap_dir, f"shap_summary_C{combo_idx+1}_class{class_idx}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"SHAP summary plot saved: {plot_path}")

    result = {
        "feature_combination": list(feature_combo),
        "confusion_matrix": cm.tolist(),
        "feature_importances": model.feature_importances_.tolist(),
        "accuracy": model.score(X_test_scaled, y_test),
        "class_labels": ["abstract", "summary-elife", "summary-plos"]
    }

    return result


def process_all_combinations(df, feature_columns, num_features, output_dir):
    all_combos = list(combinations(feature_columns, num_features))
    total = len(all_combos)
    print(f"{datetime.datetime.now()}, Running RF for all {num_features}-feature combinations...")
    print(f"--- Total combinations to process: {total}")

    results = []
    os.makedirs(output_dir, exist_ok=True)

    for idx, combo in enumerate(all_combos):
        print(f"[{idx+1}/{total}] Processing: {combo}")
        X = df[list(combo)]
        y = df['label']

        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        if X.empty or y.empty:
            print(" → Skipped due to empty data after filtering.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        result = train_and_evaluate_rf(
            X_train, X_test, y_train, y_test,
            feature_combo=combo,
            output_dir=output_dir,
            combo_idx=idx
        )
        results.append(result)

        print(f" → Accuracy: {result['accuracy']:.4f}")

    output_path = os.path.join(output_dir, f"random_forest_results_C{num_features}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"{datetime.datetime.now()}, All results saved to: {output_path}")

    if results:
        best_result = max(results, key=lambda r: r["accuracy"])
        print("\nBest performing feature combination:")
        print(f" → Features: {best_result['feature_combination']}")
        print(f" → Accuracy: {best_result['accuracy']:.4f}")
    else:
        print("No valid combinations produced results.")

    return results

def main():
    input_csv = "pre_test/formatted_features.csv"
    output_dir = "pre_test/results/randomforest"
    num_features = 14

    print("=== Random Forest Classification Analysis ===")

    df, feature_columns = load_and_prepare_data(input_csv)
    results = process_all_combinations(df, feature_columns, num_features, output_dir)

    print("\n=== Analysis Complete ===")
    print(f"Total combinations evaluated: {len(results)}")
    print(f"Results and SHAP plots available at: {output_dir}")

if __name__ == "__main__":
    main()