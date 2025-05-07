import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import numpy as np
from sklearn.metrics import classification_report

# === データの読み込み
def load_pubmed_rct(data_dir="PubMed_20k_RCT", split="train"):
    """
    PubMed RCT形式のファイルを読み込む
    各行は [LABEL]<tab>[SENTENCE]
    """
    file_path = os.path.join(data_dir, f"{split}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    abstracts, labels = [], []
    for line in lines:
        if line.strip() == "":
            continue
        label, sentence = line.strip().split("\t")
        abstracts.append(sentence)
        labels.append(label)
    
    return pd.DataFrame({"text": abstracts, "label": labels})

df = load_pubmed_rct(split="train")


# === ラベルを数値化
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])
label2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
id2label = {v: k for k, v in label2id.items()}


# === データを訓練・検証に分割
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label_id"], random_state=42)

# --- HuggingFace Dataset形式に変換
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


# === SciBERTモデル・トークナイザ読み込み
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# === トークナイズ処理
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_val = val_dataset.map(tokenize, batched=True)

# --- ラベル列名の変更とテンソル形式への変換
tokenized_train = tokenized_train.rename_column("label_id", "labels")
tokenized_val = tokenized_val.rename_column("label_id", "labels")
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === 訓練設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
)

# === モデルの訓練および評価
trainer.train() # train

predictions = trainer.predict(tokenized_val)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# ---モデルの保存
model.save_pretrained("./scibert_pubmed_rct_model")
tokenizer.save_pretrained("./scibert_pubmed_rct_model")


"""
# =====================
# Step 10: 推論の例
# =====================

# モデルの読み込み
model = AutoModelForSequenceClassification.from_pretrained("./scibert_pubmed_rct_model")
tokenizer = AutoTokenizer.from_pretrained("./scibert_pubmed_rct_model")

# 推論文
sentence = "We conducted a randomized controlled trial on 200 patients."
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

# 推論実行
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits).item()

predicted_label = model.config.id2label[predicted_class_id]
print(f"Predicted label: {predicted_label}")
"""