import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from ._dataset_types import DatasetType
import random
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM


# --- Function for Contextual Word Replacement using Masked LM ---
def contextual_word_replacement_mlm(
    text, tokenizer, model, device, n_replacements=1, top_k=5, num_augnemtations=2
):
    """
    Augments text by replacing words with predictions from a Masked Language Model.

    Args:
        text (str): The input text string.
        tokenizer: Pre-trained tokenizer (e.g., BertTokenizer).
        model: Pre-trained Masked Language Model (e.g., BertForMaskedLM).
        device (torch.device): CPU or CUDA.
        n_replacements (int): The maximum number of words to attempt to replace.
        top_k (int): Consider top_k predictions for each masked word.

    Returns:
        str: The augmented text.
    """
    if not isinstance(text, str) or not text.strip():
        return text  # Return original if text is not a valid string or empty

    # Tokenize the input text using the model's tokenizer
    # We operate on tokens as produced by the specific model's tokenizer
    original_tokens = tokenizer.tokenize(text)
    if not original_tokens:
        return text  # Return original if no tokens

    augmented_tokens = list(original_tokens)  # Make a mutable copy
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
                break  # Stop if we've made enough replacements

            current_original_token_in_loop = augmented_tokens[token_idx_to_mask]

            # Avoid masking special tokens or very short tokens (often punctuation or subwords)
            if (
                current_original_token_in_loop in tokenizer.all_special_tokens
                or len(current_original_token_in_loop) <= 1
            ):
                continue

            # Create a temporary list of tokens with one token masked
            temp_masked_tokens = list(augmented_tokens)  # Fresh copy for this attempt
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
            }  # Move inputs to the correct device

            # Get model predictions
            with torch.no_grad():  # Disable gradient calculations for inference
                outputs = model(**inputs)
                predictions = outputs.logits

            # Find the index of the [MASK] token in the input_ids
            # input_ids are usually of shape (1, sequence_length)
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


def contextual_word_replacement_augmentation(train_dataset, dataset_type: DatasetType):
    output_file = "augmented_train_dataset_2.csv"
    save_path = Path(
        f"datasets/{output_file.split('.')[0]}_{dataset_type.value}.{output_file.split('.')[1]}"
    )

    # 1. Check if an augmented dataset already exists
    if os.path.exists(save_path):
        print(f"✅ Loading pre-saved augmented dataset from '{save_path}'.")
        return pd.read_csv(save_path)

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
                augmented_text_entries = contextual_word_replacement_mlm(
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

    try:
        aug_train_dataset.to_csv(save_path, index=False)
        print(f"✅ Augmented dataset successfully saved to '{save_path}'.")
    except Exception as e:
        print(f"❌ Error saving augmented dataset to '{save_path}': {e}")

    return aug_train_dataset
