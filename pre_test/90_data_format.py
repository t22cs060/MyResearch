import pandas as pd
import os

def extract_abstract_from_file(xml_path):
    txt_path = xml_path.replace(".xml", ".txt")
    full_path = os.path.join("pre_test/data/data", txt_path)

    if not os.path.isfile(full_path):
        return ""

    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Abstract" in line:
            if i + 1 < len(lines):
                abstract_line = lines[i + 1].strip()
                clean_abstract = abstract_line.replace("#@NEW_LINE#@#", "")
                return clean_abstract
    return ""

def main():
    train_path = "pre_test/data/data/train.csv"
    df = pd.read_csv(train_path)

    df = df[["id", "Paper_Journal", "Eureka_Text_Simplified", "Full_Paper_XML"]]

    df["Paper_abstract"] = df["Full_Paper_XML"].apply(extract_abstract_from_file)

    # 抽象文字数をカウント
    df["abstract_length"] = df["Paper_abstract"].apply(len)

    # ⚠️ 文字数が0の行を除外
    df = df[df["abstract_length"] >100].copy()

    # 上位10件表示
    top10 = df.sort_values(by="abstract_length", ascending=False).tail(10)
    print("\n📊 Paper_abstract 文字数が多い上位10件:")
    for idx, row in top10.iterrows():
        print(f"- id: {row['id']} / 文字数: {row['abstract_length']}")
        print("---")

    # CSV出力
    output_df = df[["id", "Paper_Journal", "Eureka_Text_Simplified", "Paper_abstract"]]
    output_path = "pre_test/data/data/processed_output.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ 空抽象を除外し保存しました: {output_path}")

if __name__ == "__main__":
    main()
