import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import warnings
from ._dataset_types import DatasetType

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration class for augmentation parameters."""

    model_name: str = "bert-large-uncased"
    num_augmentations: int = 3  # 3-5 range, default 4
    mask_probability: float = 0.15
    min_mask_tokens: int = 1
    max_mask_tokens: int = 3
    top_k: int = 50
    temperature: float = 1.0
    batch_size: int = 32
    cache_dir: str = "./.cache/"
    output_file: str = "datasets/augmented_train_dataset.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42


class AdvancedBERTAugmenter:
    """
    Advanced BERT-based contextual data augmentation system using MLM.
    Implements sophisticated masking strategies and efficient batch processing.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._setup_random_seeds()
        self._initialize_model()

    def _setup_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

    def _initialize_model(self) -> None:
        """Initialize BERT model and tokenizer with optimizations."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")

            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                do_lower_case=True,
            )

            self.model = AutoModelForMaskedLM.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.float16
                if self.config.device == "cuda"
                else torch.float32,
            )

            # Move model to device and set to evaluation mode
            self.model.to(self.config.device)
            self.model.eval()

            # Initialize MLM pipeline for efficient inference
            self.mlm_pipeline = pipeline(
                "fill-mask",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.config.device == "cuda" else -1,
                top_k=self.config.top_k,
                batch_size=self.config.batch_size,
            )

            logger.info(f"Model loaded successfully on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    @lru_cache(maxsize=1000)
    def _get_maskable_tokens(self, text: str) -> List[Tuple[int, str]]:
        """
        Get maskable token positions with caching for efficiency.
        Returns list of (position, token) tuples for content words.
        """
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)

        # Filter out special tokens, punctuation, and very short tokens
        maskable_positions = []
        for i, token in enumerate(tokens):
            if (
                len(token) > 2
                and token.isalpha()
                and token not in self.tokenizer.all_special_tokens
                and not token.startswith("##")
            ):  # Avoid subword tokens
                maskable_positions.append((i, token))

        return maskable_positions

    def _create_masked_versions(self, text: str) -> List[str]:
        """
        Create multiple masked versions of the input text using sophisticated masking strategies.
        """
        maskable_positions = self._get_maskable_tokens(text)

        if len(maskable_positions) < self.config.min_mask_tokens:
            return [text]  # Return original if not enough maskable tokens

        masked_versions = []

        for _ in range(self.config.num_augmentations):
            # Randomly select number of tokens to mask
            num_masks = random.randint(
                self.config.min_mask_tokens,
                min(self.config.max_mask_tokens, len(maskable_positions)),
            )

            # Select positions to mask (avoid clustering)
            selected_positions = random.sample(maskable_positions, num_masks)
            selected_indices = sorted([pos[0] for pos in selected_positions])

            # Create masked version
            tokens = self.tokenizer.tokenize(text)
            for idx in selected_indices:
                if idx < len(tokens):
                    tokens[idx] = self.tokenizer.mask_token

            masked_text = self.tokenizer.convert_tokens_to_string(tokens)
            masked_versions.append(masked_text)

        return masked_versions

    def _generate_predictions(self, masked_texts: List[str]) -> List[str]:
        """
        Generate predictions for masked texts using the MLM pipeline.
        Implements temperature-based sampling for diversity.
        """
        augmented_texts = []

        try:
            # Process in batches for efficiency
            for i in range(0, len(masked_texts), self.config.batch_size):
                batch = masked_texts[i : i + self.config.batch_size]

                # Get predictions from pipeline
                predictions = self.mlm_pipeline(batch)

                # Process predictions
                for pred_group in predictions:
                    if isinstance(pred_group, list) and len(pred_group) > 0:
                        # Handle multiple masks in single text
                        if isinstance(pred_group[0], list):
                            # Multiple masks - take first prediction for each mask
                            filled_text = pred_group[0][0]["sequence"]
                        else:
                            # Single mask - apply temperature sampling
                            scores = np.array([p["score"] for p in pred_group])
                            if self.config.temperature != 1.0:
                                scores = scores / self.config.temperature

                            # Softmax with temperature
                            probs = np.exp(scores) / np.sum(np.exp(scores))

                            # Sample based on probabilities
                            choice_idx = np.random.choice(len(pred_group), p=probs)
                            filled_text = pred_group[choice_idx]["sequence"]

                        augmented_texts.append(filled_text.strip())
                    else:
                        # Fallback to original if prediction fails
                        augmented_texts.append(batch[0] if batch else "")

        except Exception as e:
            logger.error(f"Error in prediction generation: {e}")
            # Return original texts as fallback
            augmented_texts = [
                text.replace(self.tokenizer.mask_token, "[MASK]")
                for text in masked_texts
            ]

        return augmented_texts

    def augment_single_text(self, text: str) -> List[str]:
        """
        Augment a single text using contextual MLM-based augmentation.
        """
        if not text or not isinstance(text, str):
            return [text] * self.config.num_augmentations

        # Clean and preprocess text
        text = re.sub(r"\s+", " ", text.strip())

        # Handle very short texts
        if len(text.split()) < 3:
            return [text] * self.config.num_augmentations

        try:
            # Create masked versions
            masked_versions = self._create_masked_versions(text)

            # Generate augmented texts
            augmented_texts = self._generate_predictions(masked_versions)

            # Ensure we have the right number of augmentations
            while len(augmented_texts) < self.config.num_augmentations:
                augmented_texts.append(text)

            # Remove duplicates while preserving order
            seen = set()
            unique_augmentations = []
            for aug_text in augmented_texts[: self.config.num_augmentations]:
                if aug_text not in seen and aug_text != text:
                    seen.add(aug_text)
                    unique_augmentations.append(aug_text)

            # Fill up to required number if needed
            while len(unique_augmentations) < self.config.num_augmentations:
                unique_augmentations.append(text)

            return unique_augmentations[: self.config.num_augmentations]

        except Exception as e:
            logger.error(f"Error augmenting text: {e}")
            # Return original text repeated as fallback
            return [text] * self.config.num_augmentations


def create_contextual_augmentation(
    train_dataset: pd.DataFrame,
    dataset_type: DatasetType, 
    config: Optional[AugmentationConfig] = None,
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Create contextual data augmentation for the entire dataset.
    """
    config = config or AugmentationConfig()

    # Check if augmented dataset already exists
    output_path = Path(f"{config.output_file.split('.')[0]}_{dataset_type.value}{config.output_file.split('.')[1]}")
    if output_path.exists():
        logger.info(f"Loading existing augmented dataset from {output_path}")
        try:
            augmented_dataset = pd.read_csv(output_path)
            logger.info(f"Loaded {len(augmented_dataset)} records from existing file")
            return augmented_dataset
        except Exception as e:
            logger.warning(f"Failed to load existing dataset: {e}. Creating new one.")

    # Validate input dataset
    if "text_input" not in train_dataset.columns:
        raise ValueError("Dataset must contain 'text_input' column")

    logger.info(f"Starting augmentation for {len(train_dataset)} records")

    # Initialize augmenter
    augmenter = AdvancedBERTAugmenter(config)

    # Prepare augmented dataset
    augmented_records = []

    # Process dataset with progress tracking
    def process_batch(batch_data: List[Tuple[int, pd.Series]]) -> List[pd.Series]:
        """Process a batch of records."""
        batch_results = []

        for idx, row in batch_data:
            try:
                original_text = str(row["text_input"])

                # Generate augmented texts
                augmented_texts = augmenter.augment_single_text(original_text)

                # Create augmented records
                for aug_text in augmented_texts:
                    new_row = row.copy()
                    new_row["text_input"] = aug_text
                    new_row["is_augmented"] = True
                    new_row["original_index"] = idx
                    batch_results.append(new_row)

                # Add original record
                original_row = row.copy()
                original_row["is_augmented"] = False
                original_row["original_index"] = idx
                batch_results.append(original_row)

            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                # Add original record as fallback
                fallback_row = row.copy()
                fallback_row["is_augmented"] = False
                fallback_row["original_index"] = idx
                batch_results.append(fallback_row)

        return batch_results

    # Process in parallel batches
    batch_size = max(1, len(train_dataset) // n_workers)
    batches = []

    for i in range(0, len(train_dataset), batch_size):
        batch = [
            (idx, row) for idx, row in train_dataset.iloc[i : i + batch_size].iterrows()
        ]
        batches.append(batch)

    # Execute parallel processing
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch): batch for batch in batches
        }

        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result()
                augmented_records.extend(batch_results)

                # Progress update
                completed_batches = len([f for f in future_to_batch if f.done()])
                logger.info(f"Completed batch {completed_batches}/{len(batches)}")

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

    # Create final augmented dataset
    augmented_dataset = pd.DataFrame(augmented_records)

    # Reset index and clean up
    augmented_dataset = augmented_dataset.reset_index(drop=True)

    # Save augmented dataset
    try:
        augmented_dataset.to_csv(output_path, index=False)
        logger.info(f"Saved augmented dataset to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

    logger.info(
        f"Augmentation completed. Original: {len(train_dataset)}, "
        f"Augmented: {len(augmented_dataset)}"
    )

    return augmented_dataset
