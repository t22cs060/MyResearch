# === モジュール読み込み ===
import json
import csv
import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import numpy as np

# === 設定 ===
#JSON_PATH = "./pre_test/data/elife/train.json"
#OUTPUT_PATH = "pre_test/f_BGcode/label_ratios.csv"
JSON_PATH = "./pre_test/data/plos/train.json"
OUTPUT_PATH = "pre_test/f_BGcode/label_plos_ratios.csv"

# CSV入力用
#CSV_PATH = "pre_test/data/CELLS_metadata/train_meta.csv"
#OUTPUT_PATH = "pre_test/f_BGcode/bg_cells_features.csv"
CSV_PATH = "pre_test/data/data/processed_output.csv"
OUTPUT_PATH = "pre_test/f_BGcode/bg_ea_features.csv"

MODEL_PATH = "pre_test/f_BGcode/sequential_model_15.pt"
LABEL_NAMES = ["Background", "Objective", "Methods", "Results", "Conclusions"]
ID2LABEL = {i: label for i, label in enumerate(LABEL_NAMES)}

# === モデル定義 ===
class SentenceClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_size=256, num_labels=5, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_seq = cls_embeddings.unsqueeze(1)
        lstm_out, _ = self.bilstm(cls_seq)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out.squeeze(1))
        return logits

# === 推論準備 ===
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceClassifier("bert-base-uncased").to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === ラベル予測関数 ===
def predict_labels(sentences):
    predicted_labels = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = model(**inputs)
        pred_id = torch.argmax(outputs, dim=1).item()
        predicted_labels.append(ID2LABEL[pred_id])
    return predicted_labels

# === ラベル比率算出関数 ===
def compute_label_ratios(sentences):
    predicted = predict_labels(sentences)
    total = len(predicted)
    label_counts = {label: 0 for label in LABEL_NAMES}
    for label in predicted:
        label_counts[label] += 1
    label_ratios = {label: (count / total if total > 0 else 0.0) for label, count in label_counts.items()}
    return label_ratios

def compute_label_ratios_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for idx, article in enumerate(data):
        abstract = sent_tokenize(" ".join(article.get("abstract", [])))
        summary = sent_tokenize(" ".join(article.get("summary", [])))

        abstract_ratios = compute_label_ratios(abstract)
        summary_ratios = compute_label_ratios(summary)

        entry = {
            "doc_id": article.get("id", f"doc_{idx+1}")
        }
        for label in LABEL_NAMES:
            entry[f"abstract_{label.lower()}_ratio"] = abstract_ratios[label]
            entry[f"summary_{label.lower()}_ratio"] = summary_ratios[label]
        results.append(entry)

    return results

# === 可視化関数 ===
def plot_label_ratios(results):
    doc_ids = [res["doc_id"] for res in results]
    x = np.arange(len(doc_ids))
    width = 0.13

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, label in enumerate(LABEL_NAMES):
        abstract_ratios = [res[f"abstract_{label.lower()}_ratio"] for res in results]
        summary_ratios = [res[f"summary_{label.lower()}_ratio"] for res in results]
        ax.bar(x + (i - 2.5) * width, abstract_ratios, width, label=f"Abstract-{label}", alpha=0.6)
        ax.bar(x + (i - 2.0) * width, summary_ratios, width, label=f"Summary-{label}", alpha=0.9)

    ax.set_xlabel("Document ID")
    ax.set_ylabel("Label Ratio")
    ax.set_title("Label Ratios per Document")
    ax.set_xticks(x)
    ax.set_xticklabels(doc_ids, rotation=90)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# === CSV保存関数 ===
def save_ratios_to_csv(results, filename = OUTPUT_PATH):
    if not results:
        return
    keys = results[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def compute_label_ratios_from_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        doc_id = f"doc_{idx+1}"
        abstract = sent_tokenize(str(row.get("abs_text", "")))
        summary = sent_tokenize(str(row.get("pls_text", "")))

        abstract_ratios = compute_label_ratios(abstract)
        summary_ratios = compute_label_ratios(summary)

        entry = {"doc_id": doc_id}
        for label in LABEL_NAMES:
            entry[f"abstract_{label.lower()}_ratio"] = abstract_ratios[label]
            entry[f"summary_{label.lower()}_ratio"] = summary_ratios[label]
        results.append(entry)

    return results

# === 実行 ===
if __name__ == '__main__':
    print("=== Computing Label Ratios for Abstracts and Summaries ===")

    #ratios = compute_label_ratios_from_json(JSON_PATH)

    ratios = compute_label_ratios_from_csv(CSV_PATH)

    for res in ratios:
        print(f"Document ID: {res['doc_id']}")
        for label in LABEL_NAMES:
            print(f"  Abstract {label} Ratio: {res[f'abstract_{label.lower()}_ratio']:.3f}")
            print(f"  Summary  {label} Ratio: {res[f'summary_{label.lower()}_ratio']:.3f}")
        print("-" * 40)
    save_ratios_to_csv(ratios)
    #plot_label_ratios(ratios)
