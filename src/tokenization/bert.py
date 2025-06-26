from transformers import BertTokenizer, PreTrainedTokenizer
from pathlib import Path
import logging
from typing import Union

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_bert_tokenizer(
    model_name: str = "bert-base-uncased",
) -> PreTrainedTokenizer:
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
    except OSError as e:
        logging.error(f"Could not load the tokenizer for '{model_name}'. Error: {e}")
        raise

    return tokenizer


if __name__ == "__main__":
    large_bert_tokenizer = get_bert_tokenizer(model_name="bert-large-cased")
