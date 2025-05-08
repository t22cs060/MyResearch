# pubmed_seq_classifier.py

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ---------- データロード ----------
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

# ---------- データセット ----------
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
            "input_ids": encoded["input_ids"],           # (num_sentences, max_len)
            "attention_mask": encoded["attention_mask"], # (num_sentences, max_len)
            "labels": torch.tensor(encoded_labels)       # (num_sentences,)
        }

# ---------- モデル定義 ----------
class SequentialSentenceClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=256, num_labels=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask):
        B, T, L = input_ids.shape
        input_ids = input_ids.view(B * T, L)
        attention_mask = attention_mask.view(B * T, L)

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = bert_out.last_hidden_state[:, 0, :]  # (B*T, H)
        cls_embeds = cls_embeds.view(B, T, -1)            # (B, T, H)

        lstm_out, _ = self.bilstm(cls_embeds)             # (B, T, 2H)
        logits = self.classifier(self.dropout(lstm_out))  # (B, T, C)
        return logits

# ---------- トレーニング ----------
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

# ---------- メイン処理 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ファイルパス
    data_path = "pre_test/PubMed_20k_RCT/train.txt"  # 必要に応じて変更

    # データの読み込み
    abstracts = load_pubmed_rct(data_path)
    labels = [label for abs in abstracts for label, _ in abs]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_labels = len(label_encoder.classes_)

    # トークナイザ
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # データセットとローダー
    dataset = PubMedRCTDataset(abstracts, tokenizer, label_encoder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # モデルと最適化設定
    model = SequentialSentenceClassifier(num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # トレーニング
    for epoch in range(3):
        loss = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

    # モデル保存
    torch.save(model.state_dict(), "sequential_model.pt")
    print("✅ モデル保存完了：sequential_model.pt")

if __name__ == "__main__":
    main()
