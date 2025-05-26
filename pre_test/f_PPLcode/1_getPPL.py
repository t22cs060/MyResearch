import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime

# --- BioGPT初期化
biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
biogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
biogpt_model.eval()

# --- Perplexity算出関数
def compute_ppl(text: str):
    inputs = biogpt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = biogpt_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    return torch.exp(loss).item()

# --- JSON読み込み
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"{datetime.datetime.now()}, Loading JSON: {path}")
        return json.load(f)

# --- 抽象と要約の抽出
def extract_abstract_summary(data):
    abstracts, summaries, ids = [], [], []
    for idx, article in enumerate(data):
        doc_id = article.get('id', f"doc_{idx+1}")
        abstracts.append(' '.join(article.get('abstract', [])))
        summaries.append(' '.join(article.get('summary', [])))
        ids.append(doc_id)
    return ids, abstracts, summaries

# --- メイン処理
def process_ppl(json_path, output_csv):
    data = load_json(json_path)
    ids, abstracts, summaries = extract_abstract_summary(data)

    print(f"{datetime.datetime.now()}, Computing Perplexity features...")
    rows = []
    for doc_id, abs_text, sum_text in zip(ids, abstracts, summaries):
        abs_ppl = compute_ppl(abs_text)
        sum_ppl = compute_ppl(sum_text)
        rows.append({
            "doc_id": doc_id,
            "abstract_ppl": abs_ppl,
            "summary_ppl": sum_ppl
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"{datetime.datetime.now()}, PPL CSV saved to {output_csv}")

# === 実行部 ===
if __name__ == "__main__":
    #input_json = "pre_test/data/elife/train.json"
    #output_csv = "pre_test/f_PPLcode/ppl_elife_features.csv"
    input_json = "pre_test/data/plos/train.json"
    output_csv = "pre_test/f_PPLcode/ppl_plos_features.csv"
    process_ppl(input_json, output_csv)