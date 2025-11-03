import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from datasets import load_from_disk, Dataset, load_dataset
from textSummarizer.logging import logger
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device.upper()}")

        # Load tokenizer and model
        model_ckpt = "t5-small"  # lightweight model for fast training
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
        logger.info("Model and tokenizer loaded successfully")

        # Load CNN/DailyMail dataset
        logger.info("Loading small CNN/DailyMail dataset subset...")
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

        # Use smaller subset for faster training
        dataset["train"] = dataset["train"].select(range(300))
        dataset["validation"] = dataset["validation"].select(range(60))

        # Preprocessing (tokenization)
        def preprocess_function(examples):
            inputs = [doc for doc in examples["article"]]
            targets = [t for t in examples["highlights"]]
            model_inputs = tokenizer(
                inputs, max_length=512, truncation=True
            )
            labels = tokenizer(
                targets, max_length=128, truncation=True
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(
            preprocess_function, batched=True, remove_columns=["article", "highlights", "id"]
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=10,
            weight_decay=0.01,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            remove_unused_columns=False,  # Important fix!
            logging_dir=os.path.join(self.config.root_dir, "logs"),
            load_best_model_at_end=True
        )

        # Trainer setup
        trainer = Trainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"]
        )

        # Train
        logger.info("Starting model training...")
        trainer.train()
        logger.info("Model training completed successfully")

        # Save model and tokenizer
        model_dir = os.path.join(self.config.root_dir, "t5-small-cnn-model")
        tokenizer_dir = os.path.join(self.config.root_dir, "tokenizer")

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Model saved at: {model_dir}")
        logger.info(f"Tokenizer saved at: {tokenizer_dir}")