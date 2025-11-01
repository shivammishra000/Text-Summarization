import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        try:
            input_encodings = self.tokenizer(
                example_batch['dialogue'],
                max_length=1024,
                truncation=True,
                padding="max_length"
            )

            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(
                    example_batch['summary'],
                    max_length=128,
                    truncation=True,
                    padding="max_length"
                )

            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }

        except Exception as e:
            logger.error(f"Error while tokenizing batch: {e}")
            raise e

    def convert(self):
        try:
            logger.info(f"Loading dataset from: {self.config.data_path}")
            dataset_samsum = load_from_disk(self.config.data_path)

            logger.info("Starting tokenization process...")
            dataset_samsum_pt = dataset_samsum.map(
                self.convert_examples_to_features,
                batched=True,
                batch_size=16
            )

            save_path = os.path.join(self.config.root_dir, "samsum_dataset")
            dataset_samsum_pt.save_to_disk(save_path)
            logger.info(f"Tokenized dataset saved successfully at: {save_path}")

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e