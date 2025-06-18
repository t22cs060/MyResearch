import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
import os
import datetime

# --- DataFrameからsummary or abstract関連の列を抽出する関数
def extract_target_columns(df, target):
    """
    Args: df (pandas.DataFrame): 入力df
    Returns:
        pandas.DataFrame: summary関連の列のみを含むdf
        list: 抽出された列名のリスト
    """
    summary_columns = [col for col in df.columns if target in col.lower()]
    
    # --- 存在しない場合
    if not summary_columns:
        print(f"{datetime.datetime.now()}, Column containing '{target}' not found: {list(df.columns)}")
        return None, []
    
    # --- 存在する場合に抽出
    print(f"{datetime.datetime.now()}, extraction ({len(summary_columns)}items)")
    for col in summary_columns:
        print(f"  - {col}")
    summary_df = df[['doc_id'] + summary_columns].copy()

    return summary_df, summary_columns


# --- CSVファイルから主成分分析を実行
def perform_pca_analysis(csv_path, target, output_dir=None, n_components=None):
    """
    Args:
        csv_path (str): 入力CSVファイルのパス
        output_dir (str): 出力ディレクトリ（Noneの場合は入力ファイルと同じディレクトリ）
        n_components (int): 主成分の数（Noneの場合は自動決定）
    
    Returns:
        dict: 分析結果を含む辞書
    """
    try:
        # --- CSVファイルを読み込み
        print(f"{datetime.datetime.now()}, loaded CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # --- taeget関連の列を抽出 ->  doc_idを除いた数値データのみを取得
        summary_df, summary_columns = extract_target_columns(df, target)
        numeric_data = summary_df.drop('doc_id', axis=1)
        
        # --- データの基本統計を表示
        print("\n=== データの基本統計 ===")
        print(numeric_data.describe())
        
        # --- 欠損値の処理
        print(f"{datetime.datetime.now()}, missing value:")
        missing_counts = numeric_data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count}個")
        
        if missing_counts.sum() > 0:
            print(f"{datetime.datetime.now()}, impute missing values with the mean...")
            imputer = SimpleImputer(strategy='mean')
            numeric_data_filled = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index
            )
        else:
            numeric_data_filled = numeric_data
        
        # --- データの標準化
        print(f"{datetime.datetime.now()}, data normalization")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data_filled)
        
        # --- 主成分数の決定
        if n_components is None:
            n_components = min(len(summary_columns), len(df) - 1)
        
        # --- 主成分分析の実行
        print(f"{datetime.datetime.now()}, Running a PCA（主成分数: {n_components}）...")
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns)  # 結果をdfにする
        pca_df['doc_id'] = summary_df['doc_id'].values
        
        # --- 寄与率と累積寄与率の計算
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_ratio = np.cumsum(explained_variance_ratio)
        
        """
        print(f"\n=== 主成分分析結果 ===")
        for i in range(n_components):
            print(f"PC{i+1}: 寄与率 {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
        print(f"累積寄与率: {cumulative_ratio[-1]:.4f} ({cumulative_ratio[-1]*100:.2f}%)")
        """

        # --- 出力ディレクトリの設定
        os.makedirs(output_dir, exist_ok=True) 
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        # --- 1. 主成分スコアの保存
        pca_output_path = os.path.join(output_dir, f"{base_name}_pca_scores.csv")
        pca_df.to_csv(pca_output_path, index=False)
        print(f"{datetime.datetime.now()}, 主成分スコアを保存しました: {pca_output_path}")
        
        # --- 2. 主成分負荷量の保存
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=pca_columns,
            index=summary_columns
        )
        loadings_output_path = os.path.join(output_dir, f"{base_name}_pca_loadings.csv")
        loadings_df.to_csv(loadings_output_path)
        print(f"{datetime.datetime.now()}, 主成分負荷量を保存しました: {loadings_output_path}")
        
        # --- 3. 寄与率の保存
        variance_df = pd.DataFrame({
            '主成分': pca_columns,
            '寄与率': explained_variance_ratio,
            '累積寄与率': cumulative_ratio
        })
        variance_output_path = os.path.join(output_dir, f"{base_name}_pca_variance.csv")
        variance_df.to_csv(variance_output_path, index=False)
        print(f"{datetime.datetime.now()}, 寄与率を保存しました: {variance_output_path}")
        
        # 可視化
        create_pca_plots(pca_df, explained_variance_ratio, loadings_df, output_dir, base_name)
        
        return {
            'pca_scores': pca_df,
            'loadings': loadings_df,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_ratio': cumulative_ratio,
            'pca_model': pca,
            'scaler': scaler
        }
        
    except Exception as e:
        print(f"{datetime.datetime.now()}, ERROR: failure PCA")
        return None

# 主成分分析の結果を可視化する
def create_pca_plots(pca_df, explained_variance_ratio, loadings_df, output_dir, base_name):

    plt.style.use('default')
    
    # 日本語フォントの設定（可能な場合）
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass
    
    # 1. 寄与率の棒グラフ
    plt.figure(figsize=(10, 6))
    pc_labels = [f'PC{i+1}' for i in range(len(explained_variance_ratio))]
    plt.bar(pc_labels, explained_variance_ratio * 100)
    plt.title('Contribution Ratio of Each Principal Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Contribution Ratio (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_contribution_ratio.png"), dpi=300)
    plt.close()
    
    # 2. 累積寄与率のグラフ
    plt.figure(figsize=(10, 6))
    cumulative_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio * 100, 'bo-')
    plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80%')
    plt.axhline(y=90, color='g', linestyle='--', alpha=0.7, label='90%')
    plt.title('Cumulative Contribution Ratio')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Contribution Ratio (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_cumulative_ratio.png"), dpi=300)
    plt.close()
    
    # 3. 第1主成分vs第2主成分の散布図
    if len(pca_df.columns) >= 3:  # PC1, PC2, doc_idがある場合
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
        plt.title('PCA Score Plot (PC1 vs PC2)')
        plt.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_score_plot.png"), dpi=300)
        plt.close()
    
    # 4. 主成分負荷量のヒートマップ
    if len(loadings_df.columns) >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings_df.iloc[:, :min(5, len(loadings_df.columns))], 
                   annot=True, cmap='RdBu_r', center=0, fmt='.3f')
        plt.title('PCA Loadings Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_loadings_heatmap.png"), dpi=300)
        plt.close()
    
    print(f"{datetime.datetime.now()}, グラフを保存しました: {output_dir}")

# --- main func
def main():  
    #csv_path = "pre_test/marged_elife_features.csv"
    #output_dir = "pre_test/results/PCA/elife"
    #csv_path = "pre_test/marged_plos_features.csv"
    #output_dir = "pre_test/results/PCA/plos"
    #csv_path = "pre_test/marged_features.csv"
    #output_dir = "pre_test/results/PCA/abst"
    #csv_path = "pre_test/marged_cells_features.csv"
    #output_dir = "pre_test/results/PCA/cells/summ"
    csv_path = "pre_test/marged_ea_features.csv"
    output_dir = "pre_test/results/PCA/ea/abst"
    target = "abstract"
    #target = "summary"
    n_components = 10 # 主成分数

    result = perform_pca_analysis(csv_path, target, output_dir, n_components)
    
    if result:
        print(f"{datetime.datetime.now()}, completed PCA")
    else:
        print(f"{datetime.datetime.now()}, failure PCA")
        sys.exit(1)

if __name__ == "__main__":
    main()