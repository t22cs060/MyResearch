import json
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import stanza
import textstat # For acquisition of FKGL, FRE
import spacy    # For parsing grammar-dependent labels
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# === 初期化
spacy_nlp = spacy.load("en_core_web_sm")
stanza.download('en')

# === 特徴量抽出のクラス定義
class FeatureExtractor:
    def __init__(self, text: str):
        self.text = text
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text)
        self.length = len(self.words)

    def get_basic_features(self):
        avg_sentence_length = self.length / len(self.sentences) if self.sentences else 0
        fkgl = textstat.flesch_kincaid_grade(self.text)
        fre = textstat.flesch_reading_ease(self.text)
        avg_clauses = self._calculate_avg_clauses()
        return {
            'length': self.length,
            'avg_sentence_length': avg_sentence_length,
            'FKGL': fkgl,
            'FRE': fre,
            'avg_clauses': avg_clauses
        }

    def _calculate_avg_clauses(self):
        total_clauses = 0
        for sent in spacy_nlp(self.text).sents:
            total_clauses += sum(1 for token in sent if token.dep_ in ['advcl', 'ccomp', 'xcomp', 'relcl', 'conj'])
        return total_clauses / len(self.sentences) if self.sentences else 0
    
# === JSON読み込み
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        print(f"{datetime.datetime.now()}, JSON loaded: {path}")
        return json.load(f)


# === function to et abstracts and summaries
def extract_doc_texts(data):
    ids, abstracts, summaries = [], [], []
    for idx, article in enumerate(data):
        doc_id = article.get("id", f"doc_{idx+1}")
        ids.append(doc_id)
        abstracts.append(' '.join(article.get('abstract', [])))
        summaries.append(' '.join(article.get('summary', [])))
    print(f"{datetime.datetime.now()}, Abstracts and summaries extracted")
    return ids, abstracts, summaries


# === 特徴量抽出・構造化
def extract_features_per_doc(doc_ids, abstracts, summaries):
    print(f"{datetime.datetime.now()}, Extracting features")
    rows = []
    for doc_id, a_text, s_text in zip(doc_ids, abstracts, summaries):
        af = FeatureExtractor(a_text).get_basic_features()
        sf = FeatureExtractor(s_text).get_basic_features()
        row = {'doc_id': doc_id}
        for k, v in af.items():
            row[f'abstract_{k}'] = v
        for k, v in sf.items():
            row[f'summary_{k}'] = v
        rows.append(row)
    print(f"{datetime.datetime.now()}, Done")
    return rows

# === 特徴量ごとにプロット
def results_prot(df):
    features = ['length', 'avg_sentence_length', 'FKGL', 'FRE', 'avg_clauses']

    for feat in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feat, hue='type', kde=True, stat='density', common_norm=False)
        plt.title(f'Distribution of {feat}')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f"pre_test/f_BASECcode/figure/{feat}_distribution.png") 
        plt.close() 
    return df


# === 汎用処理エントリポイント
def process_dataset(json_path, output_csv):
    data = load_json(json_path)
    doc_ids, abstracts, summaries = extract_doc_texts(data)
    records = extract_features_per_doc(doc_ids, abstracts, summaries)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"{datetime.datetime.now()}, Data saved: {output_csv}")

    #_, df = load_csv_as_records(output_csv) # 保存内容を読み込む
    #results_prot(df)                        # プロット


if __name__ == '__main__':
    #input_json = "pre_test/data/elife/train.json"
    #output_csv = "pre_test/f_BASECcode/basic_elife_features.csv"
    input_json = "pre_test/data/plos/train.json"
    output_csv = "pre_test/f_BASECcode/basic_plos_features.csv"

    print(f"{datetime.datetime.now()}, START")
    process_dataset(input_json, output_csv)
    print(f"{datetime.datetime.now()}, DONE")
