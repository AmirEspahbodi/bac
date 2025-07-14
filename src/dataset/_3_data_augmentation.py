import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from ._dataset_types import DatasetType
import random
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Tuple
import logging
from tqdm.auto import tqdm
import pandas as pd
import torch

# --- Configuration ---
class _Config:
    """
    Configuration class for the data augmentation pipeline.
    """
    # Model identifiers (replace with the path to your local models on Kaggle)
    LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"

    # Generation & Augmentation Parameters
    MIN_SAMPLES_THRESHOLD = 5
    BOOTSTRAP_TARGET_COUNT = 15
    GENERATION_BATCH_SIZE = 4
    MAX_NEW_TOKENS = 150

    # SALAD Augmentation Parameters
    NON_CAUSAL_POS_TAGS = {"ADP", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "PART", "PRON", "SCONJ"}
    MASK_TOKEN = "[MASK]"


# --- Data Augmentation Pipeline ---
class _DataAugmentationPipeline:
    """
    A fully automated pipeline to address data scarcity and imbalance using
    data-driven few-shot prompting and augmentation.
    """
    def __init__(self, config: _Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._setup_logging()
        self._load_models()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline initialized on device: {self.device}")

    def _load_models(self):
        self.logger.info("Loading models...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.LLM_MODEL_ID, quantization_config=quantization_config, device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL_ID, device=self.device)
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        self.logger.info("All models loaded successfully.")

    def _generate_text_with_llm(self, prompts: List[str]) -> List[str]:
        """Generates text in batches using the local LLM."""
        generated_texts = []
        if not prompts:
            return []
        
        for i in tqdm(range(0, len(prompts), self.config.GENERATION_BATCH_SIZE), desc="LLM Generation"):
            batch_prompts = prompts[i:i + self.config.GENERATION_BATCH_SIZE]
            templated_prompts = [f"[INST] {p} [/INST]" for p in batch_prompts]
            inputs = self.tokenizer(
                templated_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs, max_new_tokens=self.config.MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for output in outputs:
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                response = full_text.split("[/INST]")[-1].strip()
                generated_texts.append(response)

        return generated_texts
    
    def _create_class_prototypes(self, df: pd.DataFrame) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
        """Creates a mean embedding (prototype) for each class from its texts."""
        self.logger.info("Creating class prototype embeddings from data...")
        class_prototypes = {}
        class_samples = {}
        
        for label in df['label'].unique():
            class_texts = df[df['label'] == label]['text'].tolist()
            class_samples[label] = class_texts
            
            # Ensure there's text to encode
            if class_texts:
                embeddings = self.embedding_model.encode(
                    class_texts, convert_to_tensor=True, show_progress_bar=False
                )
                prototype = embeddings.mean(dim=0)
                class_prototypes[label] = prototype
        
        self.logger.info(f"Created {len(class_prototypes)} class prototypes.")
        return class_prototypes, class_samples

    def _bootstrap_minority_classes(self, df: pd.DataFrame, class_samples: Dict[str, List[str]]) -> pd.DataFrame:
        """Stage 1: Generate synthetic data using few-shot examples from the data itself."""
        self.logger.info("--- Stage 1: Bootstrapping Ultra-Minority Classes ---")
        class_counts = df['label'].value_counts()
        minority_classes = class_counts[class_counts < self.config.MIN_SAMPLES_THRESHOLD].index.tolist()

        if not minority_classes:
            self.logger.info("No minority classes detected. Skipping bootstrapping.")
            return pd.DataFrame()

        self.logger.info(f"Found {len(minority_classes)} minority classes: {minority_classes}")
        prompts_to_generate, labels_for_prompts = [], []

        for class_label in minority_classes:
            if class_label not in class_samples or not class_samples[class_label]:
                continue
            
            num_to_generate = self.config.BOOTSTRAP_TARGET_COUNT - class_counts.get(class_label, 0)
            if num_to_generate <= 0:
                continue

            # Use existing samples as few-shot examples
            examples = "\n".join([f"- {s}" for s in class_samples[class_label]])
            prompt_template = (
                "You are an expert bug report writer. Given the following examples of bug reports for the '{label}' team, "
                "write a new, distinct bug report that fits the same category.\n\n"
                "EXAMPLES:\n{examples}\n\n"
                "NEW BUG REPORT:"
            )
            for _ in range(num_to_generate):
                prompts_to_generate.append(prompt_template.format(label=class_label, examples=examples))
                labels_for_prompts.append(class_label)
        
        generated_texts = self._generate_text_with_llm(prompts_to_generate)
        synthetic_df = pd.DataFrame({
            'text': generated_texts, 'label': labels_for_prompts, 'source': 'synthetic_bootstrap'
        })
        self.logger.info(f"Generated {len(synthetic_df)} new samples for minority classes.")
        return synthetic_df

    def _augment_with_salad(
        self, df: pd.DataFrame, class_prototypes: Dict[str, torch.Tensor], class_samples: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Stage 2: Augment the entire dataset using SALAD-style transformations."""
        self.logger.info("--- Stage 2: Augmenting with SALAD ---")
        
        # Positive Samples
        self.logger.info("Generating positive samples (structure-aware masking)...")
        positive_texts = [
            " ".join([token.text if token.pos_ not in self.config.NON_CAUSAL_POS_TAGS else self.config.MASK_TOKEN for token in doc])
            for doc in tqdm(self.nlp.pipe(df['text']), total=len(df), desc="POS Masking")
        ]
        positive_df = pd.DataFrame({'text': positive_texts, 'label': df['label'], 'source': 'salad_positive'})

        # Negative Samples
        self.logger.info("Generating negative samples (counterfactual generation)...")
        all_labels = list(class_prototypes.keys())
        all_prototypes = torch.stack(list(class_prototypes.values()))
        
        counterfactual_prompts, counterfactual_labels = [], []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating Counterfactual Prompts"):
            original_label, original_text = row['label'], row['text']
            if original_label not in class_prototypes:
                continue

            # Find most similar class using prototype embeddings
            original_prototype = class_prototypes[original_label]
            cosine_scores = util.cos_sim(original_prototype, all_prototypes)[0]
            cosine_scores[all_labels.index(original_label)] = -1 # Exclude self
            target_label = all_labels[torch.argmax(cosine_scores).item()]
            
            # Use examples from the target class to guide the rewrite
            target_examples = "\n".join([f"- {s}" for s in random.sample(class_samples[target_label], k=min(2, len(class_samples[target_label])))])
            prompt = (
                f"Your task is to minimally rewrite the 'ORIGINAL BUG REPORT' so it becomes a valid report for the '{target_label}' team. "
                f"To guide you, here are examples for the '{target_label}' team:\n"
                f"EXAMPLES:\n{target_examples}\n\n"
                f"Now, rewrite this report:\nORIGINAL BUG REPORT:\n---\n{original_text}"
            )
            counterfactual_prompts.append(prompt)
            counterfactual_labels.append(target_label)

        generated_counterfactuals = self._generate_text_with_llm(counterfactual_prompts)
        negative_df = pd.DataFrame({'text': generated_counterfactuals, 'label': counterfactual_labels, 'source': 'salad_negative'})
        
        self.logger.info(f"Generated {len(positive_df)} positive and {len(negative_df)} negative samples.")
        return pd.concat([positive_df, negative_df], ignore_index=True)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the full data augmentation pipeline."""
        self.logger.info("Starting data augmentation pipeline...")
        df['source'] = 'original'
        
        # Create class representations from the initial data
        class_prototypes, class_samples = self._create_class_prototypes(df)
        
        # Stage 1
        synthetic_df = self._bootstrap_minority_classes(df, class_samples)
        
        # Combine original and bootstrapped data for the next stage
        df_for_salad = pd.concat([df, synthetic_df], ignore_index=True)
        
        # Update class representations to include bootstrapped data
        class_prototypes, class_samples = self._create_class_prototypes(df_for_salad)
        
        # Stage 2
        salad_df = self._augment_with_salad(df_for_salad, class_prototypes, class_samples)
        
        final_df = pd.concat([df_for_salad, salad_df], ignore_index=True)
        
        self.logger.info(f"Pipeline finished. Original dataset size: {len(df)}, Final augmented size: {len(final_df)}")
        return final_df


def _contextual_word_replacement_mlm(
    text, tokenizer, model, device, n_replacements=1, top_k=5, num_augnemtations=2
):
    if not isinstance(text, str) or not text.strip():
        return text 

    original_tokens = tokenizer.tokenize(text)
    if not original_tokens:
        return text 

    augmented_tokens = list(original_tokens) 
    replaced_count = 0

    # Create a list of indices and shuffle them to pick random words to mask
    # We only want to mask actual word tokens, not special tokens like [CLS], [SEP] initially
    # However, for simplicity in selecting indices, we'll iterate and then check.
    result = []
    for i in range(num_augnemtations):
        token_indices = list(range(len(original_tokens)))
        random.shuffle(token_indices)
        for token_idx_to_mask in token_indices:
            if replaced_count >= n_replacements:
                break 

            current_original_token_in_loop = augmented_tokens[token_idx_to_mask]

            # Avoid masking special tokens or very short tokens (often punctuation or subwords)
            if (
                current_original_token_in_loop in tokenizer.all_special_tokens
                or len(current_original_token_in_loop) <= 1
            ):
                continue

            # Create a temporary list of tokens with one token masked
            temp_masked_tokens = list(augmented_tokens)
            temp_masked_tokens[token_idx_to_mask] = tokenizer.mask_token

            # Convert the list of tokens back to a string format suitable for the tokenizer's input
            # This is important as the tokenizer expects a string, not a list of tokens, for creating input_ids
            masked_text_for_model_input = tokenizer.convert_tokens_to_string(
                temp_masked_tokens
            )

            # Prepare input for the model
            inputs = tokenizer(
                masked_text_for_model_input,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            inputs = {
                k: v.to(device) for k, v in inputs.items()
            } 

            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits

            try:
                # Squeeze to remove batch dimension if it's 1, then convert to list
                flat_input_ids = inputs["input_ids"].squeeze().tolist()
                # Handle cases where input_ids might still be a single ID (very short text)
                if not isinstance(flat_input_ids, list):
                    flat_input_ids = [flat_input_ids]
                mask_token_index_in_ids = flat_input_ids.index(tokenizer.mask_token_id)
            except ValueError:
                # If [MASK] token ID is not found (e.g., due to truncation before mask), skip this replacement attempt
                continue

            # Get the top_k predicted token IDs for the masked position
            predicted_token_ids = torch.topk(
                predictions[0, mask_token_index_in_ids], k=top_k, dim=-1
            ).indices.tolist()

            # Try to find a suitable replacement from the predictions
            replacement_made_for_this_mask = False
            for token_id in predicted_token_ids:
                replacement_token = tokenizer.decode(
                    [token_id]
                ).strip()  # Decode the token ID to a string

                # Criteria for a good replacement:
                # 1. Not empty.
                # 2. Different from the original token (case-insensitive).
                # 3. Not a subword piece (heuristic: doesn't start with '##' for BERT).
                # 4. Not a special token (e.g., [CLS], [SEP]).
                # 5. Not an unknown token.
                if (
                    replacement_token
                    and replacement_token.lower()
                    != current_original_token_in_loop.lower()
                    and not replacement_token.startswith("##")
                    and replacement_token not in tokenizer.all_special_tokens
                    and tokenizer.convert_tokens_to_ids(replacement_token)
                    != tokenizer.unk_token_id
                ):
                    augmented_tokens[token_idx_to_mask] = (
                        replacement_token  # Perform the replacement
                    )
                    replaced_count += 1
                    replacement_made_for_this_mask = True
                    break  # Move to the next token to mask if n_replacements > 1

            # If no suitable replacement was found for this mask, the original token remains.
            # We continue to try other positions if replaced_count < n_replacements.

        # Convert the list of (potentially augmented) tokens back to a single string
        augmented_text = tokenizer.convert_tokens_to_string(augmented_tokens)
        result.append(augmented_text)

    return result


def _contextual_word_replacement_augmentation(train_dataset, dataset_type: DatasetType):
    # --- CONTEXTUAL WORD REPLACEMENT AUGMENTATION ---
    print("\nStarting Contextual Word Replacement Augmentation on train_dataset...")

    # Determine device for PyTorch (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for augmentation.")

    # Load pre-trained model and tokenizer for Masked LM
    # Using 'bert-base-uncased' as a common choice.
    # You can replace 'bert-base-uncased' with other models like 'roberta-base', etc.
    model_name = "bert-large-uncased"
    tokenizer = None
    model = None

    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name)
        model.to(device)  # Move model to the selected device
        model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        print(f"Successfully loaded tokenizer and model: {model_name}")
    except Exception as e:
        print(f"Error loading Hugging Face model/tokenizer ('{model_name}'): {e}.")
        print(
            "Augmentation will be skipped. 'aug_train_dataset' will be a direct copy of 'train_dataset'."
        )
        # If model loading fails, aug_train_dataset will be a copy of the original
        # You might want to handle this more gracefully depending on the application,
        # e.g., by exiting or falling back to a simpler augmentation.

    if tokenizer and model:
        # Create the augmented dataset by first copying the original train_dataset

        print(
            f"Augmenting 'text_input' in train_dataset. Total rows: {len(train_dataset)}"
        )

        # Define the number of words to attempt to replace per text entry
        num_replacements_per_text = (
            2  # Example: try to replace up to 2 words. Adjust as needed.
        )

        augmented_texts_list = []

        # Iterate through the 'text_input' column with a progress bar
        for entry in tqdm(train_dataset.iterrows(), desc="Augmenting texts"):
            if (
                pd.isna(entry[1]["text_input"])
                or not isinstance(entry[1]["text_input"], str)
                or not entry[1]["text_input"].strip()
            ):
                continue
            try:
                # Apply the contextual word replacement function
                augmented_text_entries = _contextual_word_replacement_mlm(
                    entry[1]["text_input"],
                    tokenizer,
                    model,
                    device,
                    n_replacements=num_replacements_per_text,
                    top_k=5,
                )
                for augmented_text_entry in augmented_text_entries:
                    entry_copy = entry[1].copy()
                    entry_copy["text_input"] = augmented_text_entry
                    augmented_texts_list.append(entry_copy)

            except Exception as e:
                # Log error and fallback to original text for robustness
                print(
                    f"Error during augmentation for text: '{str(entry[1]['text_input'])[:50]}...'. Error: {e}. Using original text."
                )

        # Assign the list of augmented texts back to the DataFrame column
        augmented_df = pd.DataFrame(augmented_texts_list)
        aug_train_dataset = pd.concat([train_dataset, augmented_df], ignore_index=True)

        print("Contextual Word Replacement Augmentation complete.")
        print(f"Shape of aug_train_dataset: {aug_train_dataset.shape}")

        # Optional: Display a few examples of original vs. augmented text
        print("\n--- Example of Original vs. Augmented Text ---")
        num_examples_to_show = min(3, len(train_dataset))  # Show up to 3 examples
        if num_examples_to_show > 0:
            for i in range(num_examples_to_show):
                original_text_example = train_dataset["text_input"].iloc[i]
                augmented_text_example = aug_train_dataset["text_input"].iloc[i]

                print(f"\nExample {i + 1}:")
                print(f"Original:   {str(original_text_example)[:150]}...")
                if original_text_example != augmented_text_example and not (
                    pd.isna(original_text_example) and pd.isna(augmented_text_example)
                ):
                    print(f"Augmented:  {str(augmented_text_example)[:150]}...")
                elif pd.isna(original_text_example) and pd.isna(augmented_text_example):
                    print("Augmented:  (Original was NaN, kept as NaN)")
                else:
                    print(
                        "Augmented:  (Not changed or error occurred during this specific augmentation)"
                    )
        else:
            print("Not enough data in train_dataset to show examples.")
    else:
        # This block executes if model loading failed earlier
        print(
            "'aug_train_dataset' is a copy of 'train_dataset' as augmentation was skipped due to model loading issues."
        )
        print(f"Shape of aug_train_dataset (copy): {aug_train_dataset.shape}")

    return aug_train_dataset

def do_data_augmentation(train_dataset, dataset_type: DatasetType):
    output_file = "augmented_train_dataset.csv"
    save_path = Path(
        f"datasets/{output_file.split('.')[0]}_{dataset_type.upper()}.{output_file.split('.')[1]}"
    )

    # 1. Check if an augmented dataset already exists
    if os.path.exists(save_path):
        print(f"✅ Loading pre-saved augmented dataset from '{save_path}'.")
        return pd.read_csv(save_path)
    
    pipeline_config = _Config()
    augmentation_pipeline = _DataAugmentationPipeline(pipeline_config)
    
    llm_prompt_and_salt_augmented_df = augmentation_pipeline.run(train_dataset)

    aug_train_dataset = _contextual_word_replacement_augmentation(llm_prompt_and_salt_augmented_df, dataset_type)

    try:
        aug_train_dataset.to_csv(save_path, index=False)
        print(f"✅ Augmented dataset successfully saved to '{save_path}'.")
    except Exception as e:
        print(f"❌ Error saving augmented dataset to '{save_path}': {e}")
    
    return aug_train_dataset