import os
os.environ["HF_HOME"] = "./cache"

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import pandas as pd
from utils.data_utils import return_kmer

KMER = 3  
SEQ_MAX_LEN = 512 

model_path = Path("./model")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

def predict(sequence):
    kmer_seq = return_kmer(sequence, K=KMER)
    encodings = tokenizer.batch_encode_plus(
        [kmer_seq],
        max_length=SEQ_MAX_LEN,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

import numpy as np
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

training_data_path = Path("data/TrainingData/Trainingdata.csv")
eval_data_path = Path("data/TestData/Testdata-2.csv")

df_training = pd.read_csv(training_data_path)

train_kmers, labels_train = [], []
for seq, label in tqdm(zip(df_training["SEQ"], df_training["CLASS"]), total=len(df_training["SEQ"])):
    kmer_seq = return_kmer(seq, K=KMER)
    train_kmers.append(kmer_seq)
    labels_train.append(label - 1)

NUM_CLASSES = len(np.unique(labels_train))


model_config = {
    "model_path": f"zhihan1996/DNA_bert_{KMER}",
    "num_classes": NUM_CLASSES,
}

df_val = pd.read_csv(eval_data_path)

val_kmers, labels_val = [], []
for seq, label in zip(df_val["SEQ"], df_val["CLASS"]):
    kmer_seq = return_kmer(seq, K=KMER)
    val_kmers.append(kmer_seq)
    labels_val.append(label - 1)

val_encodings = tokenizer.batch_encode_plus(
    val_kmers,
    max_length=SEQ_MAX_LEN,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
val_dataset = HF_dataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], labels_val
)

results_dir = Path("./model")
results_dir.mkdir(parents=True, exist_ok=True)
EPOCHS = 10
BATCH_SIZE = 8

training_args = TrainingArguments(
    output_dir=results_dir / "checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=60,
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=val_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

import importlib
from utils import data_utils

importlib.reload(data_utils)
import json
for ratio in [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]:
    for permute_sequence in ['Deletion', 'Insertion', 'Substitution', 'Duplication', 'Combined' ]:
        from utils.data_utils import return_kmer, val_dataset_generator, HF_dataset

        eval_results = []
        for val_dataset in val_dataset_generator(
            tokenizer, kmer_size=KMER, val_dir="data/TestData/", perturbation=permute_sequence, ratio=ratio
        ):
            res = trainer.evaluate(val_dataset)
            eval_results.append(res)

        avg_acc = np.mean([res["eval_accuracy"] for res in eval_results])
        avg_f1 = np.mean([res["eval_f1"] for res in eval_results])
        save_to = "results.jsonl"
        with open(save_to, "a") as f:
            f.write(json.dumps({"ratio": ratio, "permute_sequence": permute_sequence, "avg_acc": avg_acc, "avg_f1": avg_f1}) + "\n")
