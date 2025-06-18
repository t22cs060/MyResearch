import pandas as pd
import sys
import os
from functools import reduce
import datetime

# --- 複数のCSVファイルをdoc_id列をキーとして結合し、新しいCSVファイルとして保存する
def merge_csv_files(file_paths, output_path):
    """    
    Args:
        file_paths (list): 結合するCSVファイルのパスのリスト
        output_path (str): 出力するCSVファイルのパス
    """
    try:
        # 全てのファイルが存在することを確認
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"エラー: ファイル '{file_path}' が見つかりません。")
                return False
        
        # CSVファイルを読み込み
        dataframes = []
        for i, file_path in enumerate(file_paths):
            try:
                df = pd.read_csv(file_path)
                
                if 'doc_id' not in df.columns:
                    print(f"エラー: ファイル '{file_path}' にdoc_id列が見つかりません。")
                    return False
                if df['doc_id'].duplicated().any():
                    print(f"警告: ファイル '{file_path}' に重複するdoc_idがあります。")
                
                dataframes.append(df)
                print(f"{datetime.datetime.now()}, loaded: '{file_path}' ({len(df)}lines)")
                
            except Exception as e:
                print(f"エラー: ファイル '{file_path}' の読み込みに失敗しました。{str(e)}")
                return False
        
        # データフレームを結合
        print(f"{datetime.datetime.now()}, Combining data...")
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='doc_id', how='outer'), dataframes)
        
        # 結果の確認
        print(f"new data: {len(merged_df)}lines、{len(merged_df.columns)}row")
        print(f"new headers: {list(merged_df.columns)}")
        
        # 欠損値の確認
        missing_values = merged_df.isnull().sum()
        if missing_values.sum() > 0:
            print("欠損値が見つかりました:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  {col}: {count}個")
        
        # CSVファイルとして保存
        merged_df.to_csv(output_path, index=False)
        print(f"{datetime.datetime.now()}, save :{output_path}")
        return True
        
    except Exception as e:
        print(f"{datetime.datetime.now()}, ERROR: データの結合に失敗しました。{str(e)}")
        return False
        

# --- 縦方向に結合する関数
def concatenate_csv_simple(file_paths, output_path):
    """
    Args:
        file_paths (list): 結合するCSVファイルのパスのリスト
        output_path (str): 出力するCSVファイルのパス
    
    Returns:
        pandas.DataFrame: 結合されたデータフレーム
    """
    try:
        dataframes = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{datetime.datetime.now()}, ERROR: '{file_path}' is not exist")
            
            df = pd.read_csv(file_path)
            dataframes.append(df)
            print(f"{datetime.datetime.now()}, loaded: {file_path} ({len(df)}行)")
        
        # 縦方向に結合
        result_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # 保存
        result_df.to_csv(output_path, index=False)
        print(f"{datetime.datetime.now()}, done: {len(result_df)}行 → {output_path}")
        
        return result_df
        
    except Exception as e:
        print(f"{datetime.datetime.now()}, ERROR: {str(e)}")
        return None

# --- CSVファイルを指定された形式に整形する
def reshape_csv(input_file, output_file):
    """
    Parameters:
    input_file (str): 入力CSVファイルのパス
    output_file (str): 出力CSVファイルのパス
    """
    # CSVファイルを読み込み
    df = pd.read_csv(input_file)
    
    # 結果を格納するリスト
    result_rows = []
    
    # 各行を処理
    for _, row in df.iterrows():
        doc_id = row['doc_id']
        
        # abstractタイプの行を作成
        abstract_row = {
            'doc_id': doc_id,
            'type': 'abstract',
            'background_ratio': row['abstract_background_ratio'],
            'objective_ratio': row['abstract_objective_ratio']
        }
        result_rows.append(abstract_row)
        
        # summaryタイプの行を作成
        summary_row = {
            'doc_id': doc_id,
            'type': 'summary',
            'background_ratio': row['summary_background_ratio'],
            'objective_ratio': row['summary_objective_ratio']
        }
        result_rows.append(summary_row)
    
    # 結果をDataFrameに変換
    result_df = pd.DataFrame(result_rows)
    
    # CSVファイルに出力
    result_df.to_csv(output_file, index=False)
    
    print(f"整形完了: {len(result_rows)}行のデータを{output_file}に出力しました")
    return result_df
   


# --- main func
def main():
    """
    input_files = [
        "pre_test/f_BASECcode/basic_elife_features.csv",
        "pre_test/f_Coherence/pdtb_elife_features.csv",
        "pre_test/f_CWEcode/cwe_elife_features.csv",
        "pre_test/f_BGcode/label_elife_ratios.csv",
        "pre_test/f_PPLcode/ppl_elife_features.csv",
        ]
    output_file = "pre_test/marged_elife_features.csv"

    input_files = [
        "pre_test/f_BASECcode/basic_plos_features.csv",
        "pre_test/f_Coherence/pdtb_plos_features.csv",
        "pre_test/f_CWEcode/cwe_plos_features.csv",
        "pre_test/f_BGcode/label_plos_ratios.csv",
        "pre_test/f_PPLcode/ppl_plos_features.csv",
        ] 
    output_file = "pre_test/marged_plos_features.csv"

    input_files = [
        "pre_test/f_BASECcode/basic_cells_features.csv",
        "pre_test/f_Coherence/pdtb_cells_features.csv",
        "pre_test/f_CWEcode/cwe_cells_features.csv",
        "pre_test/f_BGcode/bg_cells_features.csv",
        "pre_test/f_PPLcode/ppl_cells_features.csv",
        ] 
    output_file = "pre_test/marged_cells_features.csv"
    """

    input_files = [
        "pre_test/f_BASECcode/basic_ea_features.csv",
        "pre_test/f_Coherence/pdtb_ea_features.csv",
        "pre_test/f_CWEcode/cwe_ea_features.csv",
        "pre_test/f_BGcode/bg_ea_features.csv",
        "pre_test/f_PPLcode/ppl_ea_features.csv",
        ] 
    output_file = "pre_test/marged_ea_features.csv"

    """
    input_files = ["pre_test/marged_elife_features.csv", "pre_test/marged_plos_features.csv"]
    output_file = "pre_test/marged_features.csv"

    input_file = "input_data.csv"  # 入力ファイル名
    output_file = "output_data.csv"  # 出力ファイル名
    """

    print(f"{datetime.datetime.now()}, input files: {input_files}")
    merge_csv_files(input_files, output_file) # idによる結合
    #concatenate_csv_simple(input_files, output_file) # 縦結合


if __name__ == "__main__":
    main()
