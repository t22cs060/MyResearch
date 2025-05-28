import pandas as pd
import re

def reshape_csv_flexible(input_file, output_file):
    """
    CSVファイルを柔軟に整形する（abstract_とsummary_のプレフィックスを自動検出）
    
    Parameters:
    input_file (str): 入力CSVファイルのパス
    output_file (str): 出力CSVファイルのパス
    """
    # CSVファイルを読み込み
    df = pd.read_csv(input_file)
    
    # abstract_とsummary_の列を自動検出
    abstract_cols = [col for col in df.columns if col.startswith('abstract_')]
    summary_cols = [col for col in df.columns if col.startswith('summary_')]
    
    # プレフィックスを除いた共通の列名を取得
    abstract_features = [col.replace('abstract_', '') for col in abstract_cols]
    summary_features = [col.replace('summary_', '') for col in summary_cols]
    
    # 共通する特徴量を取得
    common_features = list(set(abstract_features) & set(summary_features))
    common_features.sort()  # 順序を統一
    
    print(f"検出された共通特徴量: {common_features}")
    print(f"Abstract列数: {len(abstract_cols)}")
    print(f"Summary列数: {len(summary_cols)}")
    
    # その他の列（doc_idなど）を取得
    other_cols = [col for col in df.columns if not col.startswith('abstract_') and not col.startswith('summary_')]
    
    # 結果を格納するリスト
    result_rows = []
    
    # 各行を処理
    for _, row in df.iterrows():
        # その他の列の値を取得
        base_data = {col: row[col] for col in other_cols}
        
        # doc_idに基づいてsummaryのタイプを決定
        doc_id = str(row['doc_id']).lower()
        if 'elife' in doc_id:
            summary_type = 'summary-elife'
        elif 'journal' in doc_id:
            summary_type = 'summary-plos'
        else:
            summary_type = 'summary'
        
        # abstractタイプの行を作成
        abstract_row = base_data.copy()
        abstract_row['type'] = 'abstract'
        for feature in common_features:
            abstract_row[feature] = row[f'abstract_{feature}']
        result_rows.append(abstract_row)
        
        # summaryタイプの行を作成
        summary_row = base_data.copy()
        summary_row['type'] = summary_type
        for feature in common_features:
            summary_row[feature] = row[f'summary_{feature}']
        result_rows.append(summary_row)
    
    # 結果をDataFrameに変換
    # 列の順序を整理：その他の列 → type → 特徴量列
    column_order = other_cols + ['type'] + common_features
    result_df = pd.DataFrame(result_rows, columns=column_order)
    
    # CSVファイルに出力
    result_df.to_csv(output_file, index=False)
    
    print(f"整形完了: {len(result_rows)}行のデータを{output_file}に出力しました")
    print(f"出力列: {list(result_df.columns)}")
    
    return result_df

def analyze_csv_structure(input_file):
    """
    CSVファイルの構造を分析する
    
    Parameters:
    input_file (str): 入力CSVファイルのパス
    """
    df = pd.read_csv(input_file)
    
    print("=== CSV構造分析 ===")
    print(f"総列数: {len(df.columns)}")
    print(f"総行数: {len(df)}")
    
    # 列をカテゴリ別に分類
    abstract_cols = [col for col in df.columns if col.startswith('abstract_')]
    summary_cols = [col for col in df.columns if col.startswith('summary_')]
    other_cols = [col for col in df.columns if not col.startswith('abstract_') and not col.startswith('summary_')]
    
    print(f"\nAbstract列 ({len(abstract_cols)}個):")
    for col in sorted(abstract_cols):
        print(f"  - {col}")
    
    print(f"\nSummary列 ({len(summary_cols)}個):")
    for col in sorted(summary_cols):
        print(f"  - {col}")
    
    print(f"\nその他の列 ({len(other_cols)}個):")
    for col in other_cols:
        print(f"  - {col}")
    
    # 共通特徴量を確認
    abstract_features = [col.replace('abstract_', '') for col in abstract_cols]
    summary_features = [col.replace('summary_', '') for col in summary_cols]
    common_features = list(set(abstract_features) & set(summary_features))
    
    print(f"\n共通特徴量 ({len(common_features)}個):")
    for feature in sorted(common_features):
        print(f"  - {feature}")
    
    # AbstractまたはSummaryにのみ存在する特徴量
    abstract_only = set(abstract_features) - set(summary_features)
    summary_only = set(summary_features) - set(abstract_features)
    
    if abstract_only:
        print(f"\nAbstractのみの特徴量 ({len(abstract_only)}個):")
        for feature in sorted(abstract_only):
            print(f"  - abstract_{feature}")
    
    if summary_only:
        print(f"\nSummaryのみの特徴量 ({len(summary_only)}個):")
        for feature in sorted(summary_only):
            print(f"  - summary_{feature}")

# 使用例
if __name__ == "__main__":
    # ファイルパスを指定
    input_file = "pre_test/marged_features.csv"  # 入力ファイル名
    output_file = "pre_test/formatted_features.csv"  # 出力ファイル名
    
    try:
        # まずCSV構造を分析
        print("CSVファイルの構造を分析中...")
        analyze_csv_structure(input_file)
        
        print("\n" + "="*50)
        
        # 整形実行
        print("CSVファイルを整形中...")
        reshaped_data = reshape_csv_flexible(input_file, output_file)
        
        print("\n整形後のデータ（最初の5行）:")
        print(reshaped_data.head())
        
        print(f"\n整形前: {reshaped_data.shape[0]//2} 行")
        print(f"整形後: {reshaped_data.shape[0]} 行")
        
    except FileNotFoundError:
        print(f"エラー: {input_file}が見つかりません")
    except Exception as e:
        print(f"エラーが発生しました: {e}")