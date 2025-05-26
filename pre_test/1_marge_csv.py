import pandas as pd
import sys
import os
from functools import reduce
import datetime

def merge_csv_files(file_paths, output_path):
    """
    複数のCSVファイルをdoc_id列をキーとして結合し、新しいCSVファイルとして保存する
    
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
    
    output_file = "pre_test/marged_elife_fetures.csv"
    """
    input_files = [
        "pre_test/f_BASECcode/basic_plos_features.csv",
        "pre_test/f_Coherence/pdtb_plos_features.csv",
        "pre_test/f_CWEcode/cwe_plos_features.csv",
        "pre_test/f_BGcode/label_plos_ratios.csv",
        "pre_test/f_PPLcode/ppl_plos_features.csv",
        ]
    
    output_file = "pre_test/marged_plos_fetures.csv"
    
    print(f"{datetime.datetime.now()}, input files: {input_files}")
    merge_csv_files(input_files, output_file)


if __name__ == "__main__":
    main()