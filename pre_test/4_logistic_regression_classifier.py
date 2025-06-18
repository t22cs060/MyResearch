import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from itertools import combinations
import json
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(input_csv):
    """CSVファイルを読み込み、データを前処理する"""
    print("Loading data from CSV file...")
    df = pd.read_csv(input_csv)
    
    # typeカラムを分類ラベルとして使用
    # abstract, summary-elife, summary-plosの3クラス
    df['label'] = df['type'].map({
        'abstract': 0,
        'summary': 1
        #'summary-elife': 1, 
        #'summary-plos': 2
    })
    
    # 特徴量カラムを抽出（doc_id, type, labelを除く）
    feature_columns = [col for col in df.columns if col not in ['doc_id', 'type', 'label']]
    """
    if 'length' in feature_columns:
        feature_columns.remove('length')
    """
    print(f"Number of samples: {len(df)}")
    print(f"Feature columns: {feature_columns}")
    print(f"Class distribution:")
    print(df['type'].value_counts())
    
    return df, feature_columns

def train_logistic_regression(X_train, X_test, y_train, y_test, feature_combo):
    """ロジスティック回帰モデルを訓練し、評価する"""
    # 特徴量を標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ロジスティック回帰モデルを訓練
    model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
    model.fit(X_train_scaled, y_train)
    
    # 予測
    y_pred = model.predict(X_test_scaled)
    
    # 混同行列を計算
    cm = confusion_matrix(y_test, y_pred)
    
    # モデル係数を取得
    coefficients = model.coef_.tolist()
    intercept = model.intercept_.tolist()
    
    # 結果を辞書形式で返す
    result = {
        'feature_combination': list(feature_combo),
        'confusion_matrix': cm.tolist(),
        'model_coefficients': coefficients,
        'model_intercept': intercept,
        'accuracy': model.score(X_test_scaled, y_test),
        'class_labels': ['abstract', 'summary-elife', 'summary-plos']
    }
    
    return result

def run_all_combinations(df, comb, feature_columns, output_dir):
    """すべての特徴量の4つ組み合わせでロジスティック回帰を実行"""
    print(f"Running logistic regression for all 4-feature combinations...")
    print(f"Total number of combinations: {len(list(combinations(feature_columns, comb)))}")
    
    # 結果を格納するリスト
    all_results = []
    
    # すべての4つの特徴量組み合わせを生成
    for i, feature_combo in enumerate(combinations(feature_columns, comb)):
        print(f"Processing combination {i+1}: {feature_combo}")
        
        # 特徴量とラベルを抽出
        X = df[list(feature_combo)]
        y = df['label']
        
        # 欠損値がある行を除去
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # 訓練・テストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # ロジスティック回帰を実行
        result = train_logistic_regression(X_train, X_test, y_train, y_test, feature_combo)
        all_results.append(result)
        
        print(f"Accuracy: {result['accuracy']:.4f}")
    
    # 結果をJSONファイルに保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'logistic_regression_results_C{comb}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    
    # 最も高い精度の組み合わせを表示
    best_result = max(all_results, key=lambda x: x['accuracy'])
    print(f"\nBest performing combination:")
    print(f"Features: {best_result['feature_combination']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    
    return all_results

def main():
    """メイン実行関数"""
    # ファイルパスを静的に指定
    #input_csv = "pre_test/formatted_features.csv"
    #output_dir = "pre_test/results"
    #input_csv = "pre_test/formatted_cells_features.csv"
    #output_dir = "pre_test/results/logistic"
    input_csv = "pre_test/formatted_ea_features.csv"
    output_dir = "pre_test/results/logistic/ea"
    comb = 4

    print("Starting logistic regression classification analysis...")
    
    # データを読み込み・前処理
    df, feature_columns = load_and_prepare_data(input_csv)
    
    # 特徴量組み合わせでロジスティック回帰を実行
    results = run_all_combinations(df, comb, feature_columns, output_dir)
    
    print(f"\nAnalysis completed!")
    print(f"Total combinations processed: {len(results)}")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()