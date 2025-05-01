import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
import stanza
import math
import textstat # For acquisition of FKGL, FRE
import spacy    # For parsing grammar-dependent labels
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Definition someting
train_json_path =   "./pre_test/elife/train.json"
test_json_path =    "./pre_test/elife/test.json"

# --- nltk
nltk.data.path.append('/home/kajino/nltk_data')
# --- spacy
spacy_nlp = spacy.load("en_core_web_sm")
# --- Stanza
stanza.download('en', quiet=True)
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', verbose=False)

# --- SciBERT
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model = AutoModelForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased")
scibert_model.eval()
# --- BioGPT（causal LM用）
biogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
biogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")
biogpt_model.eval()


# === 特徴量抽出のクラス定義
class FeatureExtractor:
    def __init__(self, text: str, label: str):
        self.text = text
        self.label = label
        self.sentences = sent_tokenize(text)
        self.words = word_tokenize(text)
        self.length = len(self.words)

    def get_basic_features(self):
        avg_sentence_length = self.length / len(self.sentences) if self.sentences else 0
        fkgl = textstat.flesch_kincaid_grade(self.text)
        fre = textstat.flesch_reading_ease(self.text)
        avg_clauses = self._calculate_avg_clauses()
        return {
            'type': self.label,
            'length': self.length,
            'avg_sentence_length': avg_sentence_length,
            'FKGL': fkgl,
            'FRE': fre,
            'avg_clauses': avg_clauses
        }
    
    # --- avarage length
    def _calculate_avg_clauses(self):
        total_clauses = 0
        for sent in spacy_nlp(self.text).sents:
            total_clauses += sum(1 for token in sent if token.dep_ in ['advcl', 'ccomp', 'xcomp', 'relcl', 'conj'])
        return total_clauses / len(self.sentences) if self.sentences else 0

    # --- CWE
    def compute_cwe(self):
        doc = stanza_nlp(self.text)

        # --- 内容語（名詞句）を抽出
        content_words = [word.text for sent in doc.sentences for word in sent.words if word.upos in ['NOUN', 'PROPN']]
        if not content_words:
            return None

        entropy_values = []
        for word in content_words:
            tokenized = tokenizer(word, return_tensors="pt")
            with torch.no_grad():
                outputs = scibert_model(**tokenized)
            probs = torch.softmax(outputs.logits[0], dim=-1)
            word_ids = tokenized.input_ids[0][1:-1]  # skip CLS/SEP
            log_probs = [torch.log(probs[i, wid]).item() for i, wid in enumerate(word_ids)]
            avg_entropy = -sum(log_probs) / len(log_probs)
            entropy_values.append(avg_entropy)

        return sum(entropy_values) / len(entropy_values) if entropy_values else None
    
    # --- Perplexity
    def compute_ppl(self):
        inputs = biogpt_tokenizer(self.text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = biogpt_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return math.exp(loss.item())


# === function to read json
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        print("--- json data retrieved")
        return json.load(f)


# === function to et abstracts and summaries
def sep_abstract_summary(data):
    abstracts, summaries = [], []
    for article in data:
        abstracts.append(' '.join(article.get('abstract', [])))
        summaries.append(' '.join(article.get('summary', [])))
    print("--- abstracts and summaries retrieved ")
    return abstracts, summaries


# === 特徴量を取得してlistにまとめる
def get_records(abst, summ):
    print("--- Extracting features")
    records = []
    for a, s in zip(abst, summ):
        af = FeatureExtractor(a, 'abstract')
        sf = FeatureExtractor(s, 'summary')
        a_feat = af.get_basic_features()
        s_feat = sf.get_basic_features()
        a_feat['cwe'] = af.compute_cwe()
        s_feat['cwe'] = sf.compute_cwe()
        a_feat['ppl'] = af.compute_ppl()
        s_feat['ppl'] = sf.compute_ppl()
        records.extend([a_feat, s_feat])
    print("--- Done\n")
    return records


# === CSV保存
def save_features_to_csv(records, filename='pre_test/features.csv'):
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Data saved: {filename}")


# === CSV読み込み（records + DataFrame 両対応）
def load_csv_as_records(filename):
    df = pd.read_csv(filename) # 各行を辞書に変換
    return df.to_dict(orient='records'), df


# === 特徴量ごとにプロット
def results_prot(df):
    features = ['length', 'avg_sentence_length', 'FKGL', 'FRE', 'avg_clauses','cwe', 'ppl']

    for feat in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=feat, hue='type', kde=True, stat='density', common_norm=False)
        plt.title(f'Distribution of {feat}')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(f"pre_test/figule/{feat}_distribution.png") 
        plt.close() 
    return df


# === 汎用処理エントリポイント
def process_dataset(json_path, output_csv):
    data = load_json(json_path)                         # json data取得
    abstracts, summaries = sep_abstract_summary(data)   # abst, summに分離
    records = get_records(abstracts, summaries)         # 特徴量取得
    save_features_to_csv(records, output_csv)           # 結果をcsvに保存

    _, df = load_csv_as_records(output_csv) # 保存内容を読み込む
    results_prot(df)                        # プロット


# === main function
def main():
    print("--- start of main \n")
    process_dataset(train_json_path, output_csv='pre_test/features.csv')
    print("--- done of main")


if __name__ == '__main__':
    print("=== STERT ===")
    main()
    print("=== DONE ===")
