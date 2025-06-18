import json
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import spacy

# SBERTモデル
sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# SpaCy英語モデル（PDTB関係の判定に使用）
nlp = spacy.load("en_core_web_sm")

# 明示的接続詞
PDTB_CONNECTIVES = {
    "because", "since", "as", "so", "therefore", "thus", "hence", "consequently", "as a result", "due to",
    "however", "but", "although", "though", "even though", "nevertheless", "nonetheless", "yet", "on the other hand", "in contrast",
    "and", "also", "moreover", "furthermore", "in addition", "besides", "plus",
    "then", "next", "after that", "subsequently",
    "if", "unless", "provided that", "in case", "only if",
    "when", "while", "before", "after", "as soon as", "once", "until", "since",
    "like", "as if", "as though", "than",
    "for example", "for instance", "in fact", "indeed",
    "in conclusion", "to sum up", "overall", "finally"
}
# JSON読み込み
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"{datetime.datetime.now()}, Loading JSON: {path}")
        return json.load(f)

# 抽象と要約の抽出
def extract_abstract_summary(data):
    abstracts, summaries, ids = [], [], []
    for idx, article in enumerate(data):
        doc_id = article.get('id', f"doc_{idx+1}")
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


# SBERTによる隣接文類似度の平均
def compute_avg_adjacent_similarity(text):
    sentences = [sent.text for sent in nlp(text).sents]
    if len(sentences) < 2:
        return 1.0
    embeddings = sbert_model.encode(sentences)
    sims = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        sim = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        sims.append(sim)
    return sum(sims) / len(sims)

# PDTB風接続詞カバレッジ計算（簡易的実装）
def compute_pdtb_relation_coverage(text):
    sentences = [sent.text.lower() for sent in nlp(text).sents]
    count_with_conn = sum(any(conn in sent for conn in PDTB_CONNECTIVES) for sent in sentences)
    return count_with_conn / len(sentences) if sentences else 0

# メイン処理
def process_features(json_path, output_csv):
    import numpy as np

    #data = load_json(json_path)
    #ids, abstracts, summaries = extract_abstract_summary(data)
    ids, abstracts, summaries = load_csv(json_path)

    print(f"{datetime.datetime.now()}, Computing SBERT + PDTB features...")
    rows = []
    for doc_id, abs_text, sum_text in tqdm(zip(ids, abstracts, summaries), total=len(ids)):
        abs_sim = compute_avg_adjacent_similarity(abs_text)
        sum_sim = compute_avg_adjacent_similarity(sum_text)

        abs_pdtb = compute_pdtb_relation_coverage(abs_text)
        sum_pdtb = compute_pdtb_relation_coverage(sum_text)

        rows.append({
            "doc_id": doc_id,
            "abstract_sbert_sim": abs_sim,
            "summary_sbert_sim": sum_sim,
            "abstract_pdtb_coverage": abs_pdtb,
            "summary_pdtb_coverage": sum_pdtb
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"{datetime.datetime.now()}, CSV saved to {output_csv}")

# 実行
if __name__ == "__main__":
    #input_json = "pre_test/data/elife/train.json"
    #output_csv = "pre_test/f_Coherence/pdtb_elife_features.csv"
    #input_json = "pre_test/data/plos/train.json"
    #output_csv = "pre_test/f_Coherence/pdtb_plos_features.csv"
    #input_json = "pre_test/data/CELLS_metadata/train_meta.csv"
    #output_csv = "pre_test/f_Coherence/pdtb_cells_features.csv"
    input_json = "pre_test/data/data/processed_output.csv"
    output_csv = "pre_test/f_Coherence/pdtb_ea_features.csv"
    process_features(input_json, output_csv)