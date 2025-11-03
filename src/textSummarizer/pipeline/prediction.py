from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
import logging

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

            summarizer = pipeline(
                "summarization",
                model=self.config.model_path,
                tokenizer=tokenizer
            )

            logging.info(f"Generating summary for input: {text}")
            result = summarizer(text, **gen_kwargs)[0]["summary_text"]

            return result
        
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise e
