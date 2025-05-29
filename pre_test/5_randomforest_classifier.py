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
import numpy as np

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
    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # DataFrameとして列名を保持
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_combo)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_combo)

    # モデル訓練
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    # SHAP解析と可視化
    if output_dir is not None and combo_idx is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)

            print(f"shap_values type: {type(shap_values)}")
            if isinstance(shap_values, list):
                print(f"shap_values shapes: {[sv.shape for sv in shap_values]}")
                print(f"Number of classes detected: {len(shap_values)}")
            else:
                print(f"shap_values shape: {shap_values.shape}")
            print(f"X_test_scaled shape: {X_test_scaled.shape}")
            print(f"Unique classes in y_test: {np.unique(y_test)}")

            shap_dir = os.path.join(output_dir, "shap_plots")
            os.makedirs(shap_dir, exist_ok=True)

            # クラス数を確認
            class_names = ["abstract", "summary-elife", "summary-plos"]
            
            # SHAP値の形状をチェックして処理方法を決定
            if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                # 新しい形式: (n_samples, n_features, n_classes)
                print(f"Processing 3D SHAP values with shape: {shap_values.shape}")
                n_samples, n_features, n_classes = shap_values.shape
                
                # 各クラスごとにプロット
                for class_idx in range(n_classes):
                    print(f"Creating SHAP plot for class {class_idx}: {class_names[class_idx]}")
                    
                    # クラスごとのSHAP値を抽出 (n_samples, n_features)
                    class_shap_values = shap_values[:, :, class_idx]
                    
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(class_shap_values, X_test_scaled, show=False, 
                                    feature_names=list(feature_combo))
                    plt.title(f'SHAP Summary - {class_names[class_idx]} (Combination {combo_idx+1})')
                    plot_path = os.path.join(shap_dir, f"shap_summary_C{combo_idx+1}_class{class_idx}_{class_names[class_idx]}.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"SHAP summary plot saved: {plot_path}")
                
                # 特徴量重要度の棒グラフを作成
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # 各クラスの平均絶対SHAP値を計算
                    mean_shap_values = []
                    for class_idx in range(n_classes):
                        mean_vals = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
                        mean_shap_values.append(mean_vals)
                    
                    # プロット用データ準備
                    feature_names_list = list(feature_combo)
                    x_pos = np.arange(len(feature_names_list))
                    
                    # 各クラスの棒グラフ
                    width = 0.25
                    colors = ['skyblue', 'lightcoral', 'lightgreen']
                    for i, (mean_vals, class_name) in enumerate(zip(mean_shap_values, class_names)):
                        plt.bar(x_pos + i*width, mean_vals, width, label=class_name, 
                               alpha=0.8, color=colors[i])
                    
                    plt.xlabel('Features')
                    plt.ylabel('Mean |SHAP value|')
                    plt.title(f'Feature Importance by Class (Combination {combo_idx+1})')
                    plt.xticks(x_pos + width, feature_names_list, rotation=45, ha='right')
                    plt.legend()
                    plt.tight_layout()
                    
                    plot_path = os.path.join(shap_dir, f"shap_importance_C{combo_idx+1}_by_class.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"SHAP importance plot saved: {plot_path}")
                    
                except Exception as e:
                    print(f"Failed to create importance plot: {e}")
                
                # 全クラスの統合プロット（可能であれば）
                try:
                    print("Creating multi-class SHAP plot...")
                    plt.figure(figsize=(12, 8))
                    
                    # 各クラスのSHAP値をリスト形式に変換
                    shap_values_list = [shap_values[:, :, i] for i in range(n_classes)]
                    
                    shap.summary_plot(shap_values_list, X_test_scaled, show=False, 
                                    feature_names=list(feature_combo),
                                    class_names=class_names)
                    plt.title(f'SHAP Summary - All Classes (Combination {combo_idx+1})')
                    plot_path = os.path.join(shap_dir, f"shap_summary_C{combo_idx+1}_all_classes.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"SHAP multi-class summary plot saved: {plot_path}")
                except Exception as e:
                    print(f"Failed to create multi-class SHAP plot: {e}")
                    
            elif isinstance(shap_values, list) and len(shap_values) > 1:
                print(f"Processing multi-class SHAP values for {len(shap_values)} classes")
                
                # 各クラスごとにプロット
                for class_idx, sv in enumerate(shap_values):
                    print(f"Creating SHAP plot for class {class_idx}: {class_names[class_idx]}")
                    
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(sv, X_test_scaled, show=False, 
                                    feature_names=list(feature_combo))
                    plt.title(f'SHAP Summary - {class_names[class_idx]} (Combination {combo_idx+1})')
                    plot_path = os.path.join(shap_dir, f"shap_summary_C{combo_idx+1}_class{class_idx}_{class_names[class_idx]}.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"SHAP summary plot saved: {plot_path}")
                
                # 全クラスの統合プロット
                try:
                    print("Creating multi-class SHAP plot...")
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X_test_scaled, show=False, 
                                    feature_names=list(feature_combo),
                                    class_names=class_names)
                    plt.title(f'SHAP Summary - All Classes (Combination {combo_idx+1})')
                    plot_path = os.path.join(shap_dir, f"shap_summary_C{combo_idx+1}_all_classes.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"SHAP multi-class summary plot saved: {plot_path}")
                except Exception as e:
                    print(f"Failed to create multi-class SHAP plot: {e}")
                
                # 特徴量重要度の棒グラフも作成
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # 各クラスの平均絶対SHAP値を計算
                    mean_shap_values = []
                    for sv in shap_values:
                        mean_shap_values.append(np.mean(np.abs(sv), axis=0))
                    
                    # プロット用データ準備
                    feature_names_list = list(feature_combo)
                    x_pos = np.arange(len(feature_names_list))
                    
                    # 各クラスの棒グラフ
                    width = 0.25
                    for i, (mean_vals, class_name) in enumerate(zip(mean_shap_values, class_names)):
                        plt.bar(x_pos + i*width, mean_vals, width, label=class_name, alpha=0.8)
                    
                    plt.xlabel('Features')
                    plt.ylabel('Mean |SHAP value|')
                    plt.title(f'Feature Importance by Class (Combination {combo_idx+1})')
                    plt.xticks(x_pos + width, feature_names_list, rotation=45, ha='right')
                    plt.legend()
                    plt.tight_layout()
                    
                    plot_path = os.path.join(shap_dir, f"shap_importance_C{combo_idx+1}_by_class.png")
                    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"SHAP importance plot saved: {plot_path}")
                    
                except Exception as e:
                    print(f"Failed to create importance plot: {e}")
                    
            else:
                # 二値分類または単一SHAP値の場合
                print("Processing binary classification or single SHAP values")
                plt.figure(figsize=(10, 6))
                if isinstance(shap_values, list):
                    shap_values_to_plot = shap_values[0]  # 最初の要素を使用
                else:
                    shap_values_to_plot = shap_values
                    
                shap.summary_plot(shap_values_to_plot, X_test_scaled, show=False,
                                feature_names=list(feature_combo))
                plt.title(f'SHAP Summary (Combination {combo_idx+1})')
                plot_path = os.path.join(shap_dir, f"shap_summary_C{combo_idx+1}.png")
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close()
                print(f"SHAP summary plot saved: {plot_path}")

        except Exception as e:
            print(f"SHAP analysis failed for combination {combo_idx+1}: {e}")
            import traceback
            traceback.print_exc()

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

        try:
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
            
        except Exception as e:
            print(f" → Error processing combination {idx+1}: {e}")
            continue

    output_path = os.path.join(output_dir, f"random_forest_results_C{num_features}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"{datetime.datetime.now()}, All results saved to: {output_path}")

    if results:
        best_result = max(results, key=lambda r: r["accuracy"])
        print("\nBest performing feature combination:")
        print(f" → Features: {best_result['feature_combination']}")
        print(f" → Accuracy: {best_result['accuracy']:.4f}")
        
        # Top 5の結果も表示
        sorted_results = sorted(results, key=lambda r: r["accuracy"], reverse=True)
        print("\nTop 5 combinations:")
        for i, result in enumerate(sorted_results[:5]):
            print(f" {i+1}. Accuracy: {result['accuracy']:.4f}, Features: {result['feature_combination']}")
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