import os
import pandas as pd
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                         Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Using transformers version: {transformers.__version__}")
logging.info(f"Using torch version: {torch.__version__}")
logging.info(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    logging.warning("Running on CPU, training will be slower")

# Configuration
DATA_DIR = 'data/'
MODELS = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-base-v2',
    'gpt2': 'gpt2',
    't5': 't5-small',
    'xlnet': 'xlnet-base-cased',
    'electra': 'electra-small-discriminator'
}
DATASETS = {
    'amazon_combined': {'path': 'amazon_combined_train.csv', 'num_labels': 3},
    'sentiment140': {'path': 'sentiment140_train.csv', 'num_labels': 2},
    'reddit': {'path': 'reddit_train.csv', 'num_labels': 3}
}
OUTPUT_DIR = 'models/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Load and prepare dataset
def load_dataset(dataset_name, split='train'):
    path = os.path.join(DATA_DIR, DATASETS[dataset_name]['path'].replace('train', split))
    df = pd.read_csv(path)
    # Clean text column
    df['text'] = df['text'].fillna('').astype(str)
    invalid_rows = df['text'].apply(lambda x: not isinstance(x, str) or x.strip() == '')
    if invalid_rows.sum() > 0:
        logging.warning(f"Removing {invalid_rows.sum()} invalid rows from {dataset_name}_{split}")
        df = df[~invalid_rows]
    # Validate sentiment
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    invalid_sentiment = df['sentiment'].isna()
    if invalid_sentiment.sum() > 0:
        logging.warning(f"Removing {invalid_sentiment.sum()} rows with invalid sentiment from {dataset_name}_{split}")
        df = df[~invalid_sentiment]
    dataset = Dataset.from_pandas(df[['text', 'sentiment']])
    dataset = dataset.rename_column('sentiment', 'labels')
    return dataset

# Fine-tune model
def train_model(model_name, dataset_name, output_dir):
    logging.info(f"Training {model_name} on {dataset_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=DATASETS[dataset_name]['num_labels']
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logging.info(f"Model moved to {device}")
    
    # Tokenize dataset
    train_dataset = load_dataset(dataset_name, 'train')
    val_dataset = load_dataset(dataset_name, 'val')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{model_name}_{dataset_name}"),
        evaluation_strategy='steps',
        eval_steps=2000,
        save_strategy='steps',
        save_steps=2000,
        learning_rate=2e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_dir='logs/',
        logging_steps=1000,
        fp16=torch.cuda.is_available()
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    # Train
    train_result = trainer.train()
    logging.info(f"Training completed for {model_name} on {dataset_name}")
    
    # Save final output
    with open(os.path.join(output_dir, 'final_output.txt'), 'a') as f:
        f.write(f"Training completed for {model_name} on {dataset_name}\n")
        f.write(f"{train_result.metrics}\n\n")
    
    # Save metrics
    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
        f.write(f"Validation metrics for {model_name} on {dataset_name}\n")
        f.write(f"{metrics}\n\n")
    
    # Save model and tokenizer
    trainer.save_model(os.path.join(output_dir, f"{model_name}_{dataset_name}"))
    tokenizer.save_pretrained(os.path.join(output_dir, f"{model_name}_{dataset_name}"))
    
    return trainer

if __name__ == "__main__":
    for model_name in MODELS:
        for dataset_name in DATASETS:
            try:
                trainer = train_model(model_name, dataset_name, OUTPUT_DIR)
            except Exception as e:
                logging.error(f"Error training {model_name} on {dataset_name}: {str(e)}")
