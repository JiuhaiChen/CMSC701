import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from utils.data_utils import return_kmer, val_dataset_generator, HF_dataset
from utils.model_utils import load_model, compute_metrics
from transformers import Trainer, TrainingArguments

############################################
### Reading the training and test data ####
############################################

kmer_length = 3 

training_data_file = Path("data/TrainingData/Trainingdata.csv")
validation_data_file = Path("data/TestData/Testdata-2.csv")

training_data = pd.read_csv(training_data_file)

training_kmers, training_labels = [], []
for sequence, label in zip(training_data["SEQ"], training_data["CLASS"]):
    kmer_sequence = return_kmer(sequence, K=kmer_length)
    training_kmers.append(kmer_sequence)
    training_labels.append(label - 1)

num_classes = len(np.unique(training_labels))


model_configuration = {
    "model_path": f"zhihan1996/DNA_bert_{kmer_length}",
    "num_classes": num_classes,
}

model, tokenizer, device = load_model(model_configuration, return_model=True)

max_sequence_length = 512  

training_encodings = tokenizer.batch_encode_plus(
    training_kmers,
    max_length=max_sequence_length,
    padding=True, 
    truncation=True,  
    return_attention_mask=True,
    return_tensors="pt", 
)
training_dataset = HF_dataset(
    training_encodings["input_ids"], training_encodings["attention_mask"], training_labels
)


validation_data = pd.read_csv(validation_data_file)  

validation_kmers, validation_labels = [], []
for sequence, label in zip(validation_data["SEQ"], validation_data["CLASS"]):
    kmer_sequence = return_kmer(sequence, K=kmer_length)
    validation_kmers.append(kmer_sequence)
    validation_labels.append(label - 1)


validation_encodings = tokenizer.batch_encode_plus(
    validation_kmers,
    max_length=max_sequence_length,
    padding=True,  
    truncation=True,  
    return_attention_mask=True,
    return_tensors="pt", 
)
validation_dataset = HF_dataset(
    validation_encodings["input_ids"], validation_encodings["attention_mask"], validation_labels
)

############################################
### Training and evaluating the model #####
############################################

results_directory = Path("./model")
results_directory.mkdir(parents=True, exist_ok=True)
num_epochs = 10
batch_size = 8

wandb.init(project="DNA_bert", name=model_configuration["model_path"])
wandb.config.update(model_configuration)

training_arguments = TrainingArguments(
    output_dir=results_directory / "checkpoints", 
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,  
    weight_decay=0.01, 
    logging_steps=60,  
    load_best_model_at_end=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,  
    tokenizer=tokenizer,
)

trainer.train()

model_save_path = results_directory / "model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

evaluation_results = []
for test_dataset in val_dataset_generator(
    tokenizer, kmer_size=kmer_length, val_dir="data/TestData/"
):
    result = trainer.evaluate(test_dataset)
    evaluation_results.append(result)

average_accuracy = np.mean([result["eval_accuracy"] for result in evaluation_results])
average_f1 = np.mean([result["eval_f1"] for result in evaluation_results])

print(f"Average accuracy: {average_accuracy}")
print(f"Average F1: {average_f1}")

wandb.log({"avg_acc": average_accuracy, "avg_f1": average_f1})
wandb.finish()
