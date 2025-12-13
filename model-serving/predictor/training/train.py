#!/usr/bin/env python
"""
Train BERT bucket classifier with class weights.

Uses HuggingFace Trainer with weighted CrossEntropyLoss
to handle imbalanced bin distributions.
"""

import json
import os
from collections import Counter

import evaluate
import fire
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels, num_classes, method='inverse'):
    """
    Compute class weights based on label distribution.

    Methods:
    - 'inverse': weight = total / (num_classes * count)
    - 'sqrt_inverse': weight = sqrt(total / count)
    - 'effective': effective number of samples (for long-tailed)
    - 'none': no weighting
    """
    counts = Counter(labels)
    total = len(labels)

    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1)  # Avoid division by zero
        if method == 'inverse':
            w = total / (num_classes * count)
        elif method == 'sqrt_inverse':
            w = np.sqrt(total / count)
        elif method == 'effective':
            beta = 0.9999
            effective_num = (1 - beta ** count) / (1 - beta)
            w = total / (num_classes * effective_num)
        else:
            w = 1.0
        weights.append(w)

    # Normalize weights (min weight = 1.0)
    weights = np.array(weights)
    weights = weights / weights.min()

    return weights.tolist()


def main(
    ds_dir='model-serving/predictor/training/data',
    outdir='model-serving/predictor/models',
    model_name='bert_bucket',
    num_labels=5,
    weight_method='inverse',
    epochs=5,
    batch_size=32,
    lr=2e-5,
):
    """
    Train BERT bucket classifier.

    Args:
        ds_dir: Directory with train.json and valid.json
        outdir: Output directory for model
        model_name: Model name (subdirectory)
        num_labels: Number of classification bins
        weight_method: Class weight method (inverse/sqrt_inverse/effective/none)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    """
    print(f"Loading data from {ds_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Load datasets
    train_json_path = os.path.join(ds_dir, 'train.json')
    valid_json_path = os.path.join(ds_dir, 'valid.json')

    train_set = load_dataset('json', data_files=train_json_path)['train']
    valid_set = load_dataset('json', data_files=valid_json_path)['train']

    # Tokenize
    def tokenize_function(sequence):
        return tokenizer(sequence['prompt'], padding='max_length', truncation=True)

    encoded_train = train_set.map(tokenize_function, batched=True)
    encoded_valid = valid_set.map(tokenize_function, batched=True)

    # Compute class weights
    train_labels = encoded_train['label']
    print(f"\nLabel distribution (train):")
    counts = Counter(train_labels)
    for i in sorted(counts.keys()):
        print(f"  Bin {i}: {counts[i]:>5} ({100*counts[i]/len(train_labels):.1f}%)")

    if weight_method != 'none':
        class_weights = compute_class_weights(train_labels, num_labels, weight_method)
        print(f"\nClass weights ({weight_method}):")
        for i, w in enumerate(class_weights):
            print(f"  Bin {i}: {w:.2f}")
    else:
        class_weights = None
        print("\nNo class weights (standard training)")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )

    # Metrics
    mse_metric = evaluate.load('mse')
    acc_metric = evaluate.load('accuracy')

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        mse = mse_metric.compute(predictions=predictions, references=labels)
        acc = acc_metric.compute(predictions=predictions, references=labels)

        # Per-class accuracy
        per_class_acc = {}
        for c in range(num_labels):
            mask = labels == c
            if mask.sum() > 0:
                class_acc = (predictions[mask] == labels[mask]).mean()
                per_class_acc[f'acc_bin{c}'] = class_acc

        return {**mse, **acc, **per_class_acc}

    # Training arguments
    cache_dir = os.path.join(outdir, 'cache', model_name)
    os.makedirs(cache_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=cache_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        logging_steps=100,
        report_to='none',
    )

    # Create trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_valid,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save model
    save_path = os.path.join(outdir, model_name)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"\nModel saved to {save_path}")

    # Final evaluation
    print("\nFinal evaluation:")
    results = trainer.evaluate()
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save training config
    config_path = os.path.join(save_path, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'num_labels': num_labels,
            'weight_method': weight_method,
            'class_weights': class_weights,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
        }, f, indent=2)

    print("\nDone!")
    print(f"\nTo use this model, copy bin_edges.npy to {save_path}/")
    print(f"  cp {ds_dir}/bin_edges.npy {save_path}/../")


if __name__ == '__main__':
    fire.Fire(main)
