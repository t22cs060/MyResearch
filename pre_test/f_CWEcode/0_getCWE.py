import json
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
import stanza
import datetime

# --- 初期化（SciBERT, Stanza, NLTK）
nltk_tokenizer = PunktSentenceTokenizer()
stanza.download('en')
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', verbose=False)
scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModelForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model.eval()

# --- CWE計算関数
def compute_cwe(text: str):
    doc = stanza_nlp(text)
    content_words = [word.text for sent in doc.sentences for word in sent.words if word.upos in ['NOUN', 'PROPN']]
    if not content_words:
        return None

    entropy_values = []
    for word in content_words:
        tokenized = scibert_tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = scibert_model(**tokenized)
        probs = torch.softmax(outputs.logits[0], dim=-1)
        word_ids = tokenized.input_ids[0][1:-1]  # skip CLS/SEP
        if len(word_ids) == 0: continue
        log_probs = [torch.log(probs[i, wid]).item() for i, wid in enumerate(word_ids)]
        avg_entropy = -sum(log_probs) / len(log_probs)
        entropy_values.append(avg_entropy)

    return sum(entropy_values) / len(entropy_values) if entropy_values else None

# --- JSON読み込み
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"{datetime.datetime.now()}, Loading JSON: {path}")
        return json.load(f)

# --- 抽象と要約の抽出
def extract_abstract_summary(data):
    abstracts, summaries, ids = [], [], []
    for idx, article in enumerate(data):
        doc_id = article.get("id", f"doc_{idx+1}")
        abstracts.append(' '.join(article.get('abstract', [])))
        summaries.append(' '.join(article.get('summary', [])))
        ids.append(doc_id)
    return ids, abstracts, summaries

# === CSV読み込み
def load_csv(path):
    print(f"{datetime.datetime.now()}, CSV loaded: {path}")
    df = pd.read_csv(path)
    ids = df.index.astype(str).tolist()
    abstracts = df['abs_text'].fillna('').tolist()
    summaries = df['pls_text'].fillna('').tolist()
    print(f"{datetime.datetime.now()}, Abstracts and summaries extracted from CSV")
    return ids, abstracts, summaries

# --- メイン処理
def process_cwe(json_path, output_csv):
    #data = load_json(json_path)
    #ids, abstracts, summaries = extract_abstract_summary(data)
    ids, abstracts, summaries = load_csv(json_path)

    print(f"{datetime.datetime.now()}, Computing CWE features...")
    rows = []
    for doc_id, abs_text, sum_text in zip(ids, abstracts, summaries):
        abs_cwe = compute_cwe(abs_text)
        sum_cwe = compute_cwe(sum_text)
        rows.append({
            "doc_id": doc_id,
            "abstract_cwe": abs_cwe,
            "summary_cwe": sum_cwe
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"{datetime.datetime.now()}, CWE CSV saved to {output_csv}")

# === 実行部 ===
if __name__ == "__main__":
    #input_json = "pre_test/data/elife/train.json"
    #output_csv = "pre_test/f_CWEcode/cwe_elife_features.csv"
    #input_json = "pre_test/data/plos/train.json"
    #output_csv = "pre_test/f_CWEcode/cwe_plos_features.csv"
    #input_json = "pre_test/data/CELLS_metadata/train_meta.csv"
    #output_csv = "pre_test/f_CWEcode/cwe_cells_features.csv"
    input_json = "pre_test/data/data/processed_output.csv"
    output_csv = "pre_test/f_CWEcode/cwe_ea_features.csv"
    process_cwe(input_json, output_csv)