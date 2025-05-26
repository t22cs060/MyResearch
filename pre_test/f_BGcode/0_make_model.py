import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import datetime

# ---シードの設定
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# --- データロード
def load_pubmed_rct(path):
    abstracts = []
    current_abstract = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_abstract:
                    abstracts.append(current_abstract)
                    current_abstract = []
                continue
            if "\t" in line:
                label, sentence = line.split("\t")
                current_abstract.append((label, sentence))
    if current_abstract:
        abstracts.append(current_abstract)
    return abstracts

# --- カスタム collate_fn
def collate_fn(batch):
    max_sentences = max(item["input_ids"].shape[0] for item in batch)

    def pad_tensor(tensor, pad_len, pad_value=0):
        # tensor shape: (sentences, seq_len)
        if tensor.shape[0] == pad_len:
            return tensor
        pad_size = (pad_len - tensor.shape[0],) + tensor.shape[1:]
        pad_tensor = torch.full(pad_size, pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, pad_tensor], dim=0)

    input_ids = torch.stack([
        pad_tensor(item["input_ids"], max_sentences, pad_value=0)
        for item in batch
    ])
    attention_mask = torch.stack([
        pad_tensor(item["attention_mask"], max_sentences, pad_value=0)
        for item in batch
    ])
    labels = torch.stack([
        pad_tensor(item["labels"], max_sentences, pad_value=-100)  # -100は損失計算時に無視する値
        for item in batch
    ])

    return {
        "input_ids": input_ids.long(),         # (batch, max_sentences, max_len)
        "attention_mask": attention_mask.long(),
        "labels": labels.long()
    }

# --- データ・セット
class PubMedRCTDataset(Dataset):
    def __init__(self, abstracts, tokenizer, label_encoder, max_len=128):
        self.abstracts = abstracts
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = max_len

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        sentences = [s for _, s in self.abstracts[idx]]
        labels = [l for l, _ in self.abstracts[idx]]
        encoded_labels = self.label_encoder.transform(labels)

        encoded = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(encoded_labels)
        }

# --- モデル定義
class SentenceClassifier(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", hidden_size=256, num_labels=5, dropout_rate=0.3):
        super(SentenceClassifier, self).__init__()
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
        batch_size, seq_len, max_len = input_ids.size()
        input_ids = input_ids.view(-1, max_len)
        attention_mask = attention_mask.view(-1, max_len)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_seq = cls_embeddings.view(batch_size, seq_len, -1)

        lstm_out, _ = self.bilstm(cls_seq)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

# --- 損失関数の定義
def compute_class_weights(train_labels, num_classes):
    all_labels = [label for sample in train_labels for label in sample]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=all_labels)
    return torch.tensor(class_weights, dtype=torch.float)

# --- モデル評価
def evaluate_model(model, dataloader, label_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=-1)
            mask = labels != -100

            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# --- トレーニング
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# --- テスト動作
def compute_background_ratio(model, tokenizer, abstracts, label_encoder, device):
    model.eval()
    background_label = label_encoder.transform(["BACKGROUND"])[0]
    total_sentences = 0
    background_count = 0

    with torch.no_grad():
        for abstract in abstracts:
            sentences = [s for _, s in abstract]
            inputs = tokenizer(sentences, return_tensors="pt", truncation=True,
                               padding=True, max_length=128).to(device)
            logits = model(inputs["input_ids"].unsqueeze(0), inputs["attention_mask"].unsqueeze(0))
            preds = torch.argmax(logits, dim=-1).squeeze(0)

            background_count += (preds == background_label).sum().item()
            total_sentences += len(preds)

    ratio = background_count / total_sentences if total_sentences else 0
    print(f"\U0001F50D 背景文割合: {ratio:.2%}")
    return ratio

# --- メイン処理
def main():
    print(f"{datetime.datetime.now()}, start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "pre_test/PubMed_20k_RCT/train.txt"  # ファイルパス

    # データの読み込み
    abstracts = load_pubmed_rct(data_path)
    labels = [label for abs in abstracts for label, _ in abs]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_labels = len(label_encoder.classes_)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # トークナイザ

    # --- データセットとローダー
    dataset = PubMedRCTDataset(abstracts, tokenizer, label_encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # --- モデルと最適化設定
    model = SentenceClassifier(num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(15):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"{datetime.datetime.now()}, Epoch {epoch + 1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "sequential_model_15.pt")
    print(f"{datetime.datetime.now()}, モデル保存完了：sequential_model.pt")

    compute_background_ratio(model, tokenizer, abstracts, label_encoder, device)

if __name__ == "__main__":
    main()
