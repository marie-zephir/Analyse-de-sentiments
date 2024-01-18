import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import create_optimizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback
from datasets import Dataset
import evaluate
import numpy as np

def preprocess_function(examples):
    return tokenizer(examples["Commentaire"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Charger les données d'entraînement et de validation
train_data = pd.read_csv('final2_train.csv', sep=';')
val_data = pd.read_csv('final2_dev.csv', sep=';')

train_data['label'] = train_data['label'].str.replace(',', '.').astype(float)
train_data['label'] = (train_data['label'] * 2) - 1
train_data['label'] = train_data['label'].astype(int)

val_data['label'] = val_data['label'].str.replace(',', '.').astype(float)
val_data['label'] = (val_data['label'] * 2) - 1
val_data['label'] = val_data['label'].astype(int)


train = Dataset.from_pandas(train_data)
val = Dataset.from_pandas(val_data)

print("Tokenization...")

tokenized_train = train.map(preprocess_function, batched=True)
tokenized_val = val.map(preprocess_function, batched=True)

id2label = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE", 10: "TEN"}
label2id = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10}

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_train) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

accuracy = evaluate.load("accuracy")
print("Model creation...")
print(tokenized_train.features)
print(tokenized_val.features)


model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10, id2label=id2label, label2id=label2id)
tf_train_set = model.prepare_tf_dataset(
    tokenized_train,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_val,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
print("Model compilation...")
model.compile(optimizer=optimizer)
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
callbacks = [metric_callback]
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)

print("Saving model...")
model.save('BERT.h5')