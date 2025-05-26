import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
import os

def extract_summary_columns(df):
    """
    DataFrameからsummary関連の列を抽出する
    
    Args:
        df (pandas.DataFrame): 入力データフレーム
        
    Returns:
        pandas.DataFrame: summary関連の列のみを含むデータフレーム
        list: 抽出された列名のリスト
    """
    # summary関連の列を抽出（列名に'summary'が含まれる列）
    summary_columns = [col for col in df.columns if 'summary' in col.lower()]
    
    if not summary_columns:
        print("警告: 'summary'を含む列が見つかりませんでした。")
        print(f"利用可能な列: {list(df.columns)}")
        return None, []
    
    print(f"抽出されたsummary関連の列 ({len(summary_columns)}個):")
    for col in summary_columns:
        print(f"  - {col}")
    
    summary_df = df[['doc_id'] + summary_columns].copy()
    return summary_df, summary_columns

def perform_pca_analysis(csv_path, output_dir=None, n_components=None):
    """
    CSVファイルから主成分分析を実行する
    
    Args:
        csv_path (str): 入力CSVファイルのパス
        output_dir (str): 出力ディレクトリ（Noneの場合は入力ファイルと同じディレクトリ）
        n_components (int): 主成分の数（Noneの場合は自動決定）
    
    Returns:
        dict: 分析結果を含む辞書
    """
    try:
        # CSVファイルを読み込み
        print(f"CSVファイルを読み込んでいます: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"データサイズ: {df.shape[0]}行 × {df.shape[1]}列")
        
        # summary関連の列を抽出
        summary_df, summary_columns = extract_summary_columns(df)
        if summary_df is None:
            return None
        
        # doc_idを除いた数値データのみを取得
        numeric_data = summary_df.drop('doc_id', axis=1)
        
        # データの基本統計を表示
        print("\n=== データの基本統計 ===")
        print(numeric_data.describe())
        
        # 欠損値の処理
        print(f"\n欠損値の確認:")
        missing_counts = numeric_data.isnull().sum()
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count}個")
        
        if missing_counts.sum() > 0:
            print("欠損値を平均値で補完します...")
            imputer = SimpleImputer(strategy='mean')
            numeric_data_filled = pd.DataFrame(
                imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index
            )
        else:
            numeric_data_filled = numeric_data
        
        # データの標準化
        print("データを標準化しています...")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data_filled)
        
        # 主成分数の決定
        if n_components is None:
            n_components = min(len(summary_columns), len(df) - 1)
        
        # 主成分分析の実行
        print(f"主成分分析を実行しています（主成分数: {n_components}）...")
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # 結果をDataFrameに変換
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_columns)
        pca_df['doc_id'] = summary_df['doc_id'].values
        
        # 寄与率と累積寄与率の計算
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_ratio = np.cumsum(explained_variance_ratio)
        
        print(f"\n=== 主成分分析結果 ===")
        for i in range(n_components):
            print(f"PC{i+1}: 寄与率 {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")
        print(f"累積寄与率: {cumulative_ratio[-1]:.4f} ({cumulative_ratio[-1]*100:.2f}%)")
        
        # 出力ディレクトリの設定
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 結果の保存
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        
        # 1. 主成分スコアの保存
        pca_output_path = os.path.join(output_dir, f"{base_name}_pca_scores.csv")
        pca_df.to_csv(pca_output_path, index=False)
        print(f"主成分スコアを保存しました: {pca_output_path}")
        
        # 2. 主成分負荷量の保存
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=pca_columns,
            index=summary_columns
        )
        loadings_output_path = os.path.join(output_dir, f"{base_name}_pca_loadings.csv")
        loadings_df.to_csv(loadings_output_path)
        print(f"主成分負荷量を保存しました: {loadings_output_path}")
        
        # 3. 寄与率の保存
        variance_df = pd.DataFrame({
            '主成分': pca_columns,
            '寄与率': explained_variance_ratio,
            '累積寄与率': cumulative_ratio
        })
        variance_output_path = os.path.join(output_dir, f"{base_name}_pca_variance.csv")
        variance_df.to_csv(variance_output_path, index=False)
        print(f"寄与率を保存しました: {variance_output_path}")
        
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
        print(f"エラー: 主成分分析の実行に失敗しました。{str(e)}")
        return None

def create_pca_plots(pca_df, explained_variance_ratio, loadings_df, output_dir, base_name):
    """
    主成分分析の結果を可視化する
    """
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
        
        # サンプル数が少ない場合はdoc_idを表示
        if len(pca_df) <= 20:
            for i, doc_id in enumerate(pca_df['doc_id']):
                plt.annotate(doc_id, (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
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
    
    print(f"グラフを保存しました: {output_dir}")

def main():
    """
    メイン関数
    """
    if len(sys.argv) < 2:
        print("使用方法: python pca_analysis.py <結合済みCSVファイル> [出力ディレクトリ] [主成分数]")
        print("例: python pca_analysis.py merged_data.csv output/ 5")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    n_components = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not os.path.exists(csv_path):
        print(f"エラー: ファイル '{csv_path}' が見つかりません。")
        sys.exit(1)
    
    print("=== 主成分分析ツール ===")
    result = perform_pca_analysis(csv_path, output_dir, n_components)
    
    if result:
        print("\n主成分分析が完了しました！")
        print("出力ファイル:")
        print("  - *_pca_scores.csv: 主成分スコア")
        print("  - *_pca_loadings.csv: 主成分負荷量")
        print("  - *_pca_variance.csv: 寄与率")
        print("  - *.png: 可視化グラフ")
    else:
        print("主成分分析が失敗しました。")
        sys.exit(1)

if __name__ == "__main__":
    main()