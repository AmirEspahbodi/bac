import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
import random
from tqdm.auto import tqdm # For progress bars
from pathlib import Path
from ._dataset_types import DatasetType


def _mask_tokens(input_ids_tensor: torch.Tensor, tokenizer: AutoTokenizer, mask_percentage: float) -> tuple[torch.Tensor, list[int]]:
    """
    Helper function to mask tokens for Masked Language Modeling (MLM).
    Randomly masks `mask_percentage` of non-special tokens in `input_ids_tensor`
    following BERT's masking strategy (80% [MASK], 10% random, 10% original).
    """
    # Identify non-special tokens (don't mask CLS, SEP, PAD)
    all_token_ids = input_ids_tensor[0].tolist()
    non_special_token_indices = [
        i for i, token_id in enumerate(all_token_ids)
        if token_id not in tokenizer.all_special_ids
    ]

    # Determine how many non-special tokens to mask
    num_tokens_to_mask = min(
        max(1, int(len(non_special_token_indices) * mask_percentage)), # At least 1 token, or calculated percentage
        len(non_special_token_indices) # Cannot mask more tokens than available non-special tokens
    )
    
    if num_tokens_to_mask == 0:
        return input_ids_tensor, [] # No tokens to mask, return original and empty masked indices

    # Randomly select indices of tokens to mask from the non-special tokens
    masked_indices_in_original_tensor = random.sample(non_special_token_indices, num_tokens_to_mask)

    # Apply BERT's masking strategy: 80% [MASK], 10% random, 10% original
    masked_input_ids = input_ids_tensor.clone()

    num_mask_token = int(num_tokens_to_mask * 0.8)
    num_random_token = int(num_tokens_to_mask * 0.1)
    # The remaining tokens (approx. 10%) will be kept as original, no explicit action needed for them.

    # Shuffle masked_indices to apply different strategies fairly
    random.shuffle(masked_indices_in_original_tensor)

    for i, original_idx in enumerate(masked_indices_in_original_tensor):
        if i < num_mask_token:
            # 80% - replace with [MASK] token
            masked_input_ids[0, original_idx] = tokenizer.mask_token_id
        elif i < num_mask_token + num_random_token:
            # 10% - replace with a random token ID from the vocabulary
            random_token_id = random.randint(0, tokenizer.vocab_size - 1)
            masked_input_ids[0, original_idx] = random_token_id
        # Else (remaining 10%): keep the original token, so no change to masked_input_ids is needed

    return masked_input_ids, masked_indices_in_original_tensor

def contextual_data_augmentation(train_dataset: pd.DataFrame, dataset_type:DatasetType, output_file: str = "augmented_train_dataset.csv") -> pd.DataFrame:
    """
    Performs contextual data augmentation on a textual dataset using a Masked Language Model (MLM).
    It augments the 'text_input' field, creating 3-5 new records for each original,
    and returns a combined DataFrame of original and augmented data.
    Handles long texts by augmenting a random segment.
    Checks for a pre-saved augmented dataset to avoid re-computation.
    """
    
    save_path = Path(f"datasets/{output_file.split('.')[0]}_{dataset_type.value}.{output_file.split('.')[1]}")
    
    # 1. Check if an augmented dataset already exists
    if os.path.exists(save_path):
        print(f"✅ Loading pre-saved augmented dataset from '{save_path}'.")
        return pd.read_csv(save_path)

    print("🚀 Starting contextual data augmentation process...")

    # 2. Initialize Model and Tokenizer
    # Using 'bert-base-uncased' as a robust and widely available MLM model.
    # For different languages, consider models like 'bert-base-multilingual-cased' or language-specific BERTs.
    model_name = "bert-large-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        print("Please ensure you have an active internet connection or the model is cached locally.")
        raise

    model.eval() # Set model to evaluation mode for inference (disables dropout, etc.)

    # Check for GPU availability and move model to GPU if possible for efficiency
    if torch.cuda.is_available():
        model.to('cuda')
        device = 'cuda'
        print("⚡️ Using CUDA (GPU) for faster augmentation.")
    else:
        device = 'cpu'
        print("🐌 CUDA (GPU) not available. Using CPU for augmentation.")

    # 3. Augmentation Parameters
    # The number of new augmented records to create per original record.
    num_augmentations_per_original = 2 # Randomly choose between 3 to 5 augmentations
    mask_percentage = 0.15 # Percentage of tokens to mask for MLM (e.g., 15%)
    
    # Maximum sequence length for the chosen BERT model (use model.config.max_position_embeddings for definitive value)
    max_seq_len = model.config.max_position_embeddings

    augmented_records = []

    # Iterate through each record in the original dataset with a progress bar
    # `tqdm.auto` automatically selects the best progress bar based on the environment.
    for index, row in tqdm(train_dataset.iterrows(), total=len(train_dataset), desc="Augmenting text records"):
        original_text = str(row['text_input']) # Ensure text_input is treated as a string

        # Add the original record to the list of augmented records first
        augmented_records.append(row.to_dict())

        for _ in range(num_augmentations_per_original):
            # Tokenize the full original text without adding special tokens initially,
            # as we might need to slice it for long text handling.
            full_tokenized = tokenizer(original_text, add_special_tokens=False, truncation=False, return_tensors="pt")
            full_input_ids = full_tokenized['input_ids']

            augmented_text = ""

            # Handle long texts: if the tokenized text exceeds model's max sequence length
            # (minus 2 for [CLS] and [SEP] tokens that will be added later).
            if full_input_ids.shape[1] > (max_seq_len - 2):
                # Calculate the maximum possible start index for a non-special token segment that fits.
                # A segment will be `max_seq_len - 2` tokens long.
                max_start_idx = full_input_ids.shape[1] - (max_seq_len - 2)

                # Randomly select a start token index for the segment
                # Ensure the random choice is valid (at least 0, at most max_start_idx)
                segment_start_token_idx = random.randint(0, max(0, max_start_idx))

                # Extract the token IDs for the chosen segment (without special tokens)
                raw_segment_token_ids = full_input_ids[0, segment_start_token_idx : segment_start_token_idx + (max_seq_len - 2)].tolist()

                # Convert these token IDs back to a string for re-tokenization with special tokens
                # This ensures the tokenizer handles truncation and special tokens correctly for the segment
                segment_string = tokenizer.decode(raw_segment_token_ids, skip_special_tokens=True)

                encoded_input_segment = tokenizer(
                    segment_string,
                    add_special_tokens=True,
                    max_length=max_seq_len,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                input_ids = encoded_input_segment['input_ids']
                attention_mask = encoded_input_segment['attention_mask']
                token_type_ids = encoded_input_segment['token_type_ids']

                # The `_mask_tokens` function expects a tensor of shape (1, sequence_length)
                masked_input_ids, masked_indices = _mask_tokens(input_ids, tokenizer, mask_percentage)

                # Perform prediction using the masked segment
                with torch.no_grad(): # Disable gradient calculation for inference
                    logits = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    ).logits

                # Get the predicted token IDs for the masked positions
                predicted_token_ids = torch.argmax(logits[0, masked_indices], dim=-1)

                # Create a modified version of the segment's input IDs with predictions
                modified_segment_input_ids = input_ids.clone()
                for i, masked_idx in enumerate(masked_indices):
                    modified_segment_input_ids[0, masked_idx] = predicted_token_ids[i]

                # Decode the augmented segment (excluding special tokens for clean concatenation)
                augmented_segment_text = tokenizer.decode(
                    modified_segment_input_ids[0].cpu().numpy(), skip_special_tokens=True
                )

                # Decode the original prefix and suffix parts of the text (without special tokens)
                original_prefix_text = tokenizer.decode(
                    full_input_ids[0, :segment_start_token_idx].tolist(), skip_special_tokens=True
                )
                original_suffix_text = tokenizer.decode(
                    full_input_ids[0, segment_start_token_idx + (max_seq_len - 2):].tolist(), skip_special_tokens=True
                )

                # Reconstruct the full augmented text by combining original parts and augmented segment
                # Use .strip() to clean up any extra spaces at the beginning/end from decoding.
                augmented_text = (original_prefix_text + " " + augmented_segment_text + " " + original_suffix_text).strip()

            else: # Text fits within or is shorter than the model's max_seq_len
                # Standard tokenization and masking for texts that fit.
                encoded_input = tokenizer(
                    original_text,
                    add_special_tokens=True,
                    max_length=max_seq_len, # Ensure truncation to model's actual max_position_embeddings
                    truncation=True,
                    return_tensors="pt"
                ).to(device) # Move directly to device

                input_ids = encoded_input['input_ids']
                attention_mask = encoded_input['attention_mask']
                token_type_ids = encoded_input['token_type_ids'] # For BERT, segment IDs are important

                # Mask tokens in the input
                masked_input_ids, masked_indices = _mask_tokens(input_ids, tokenizer, mask_percentage)

                # Perform prediction
                with torch.no_grad():
                    logits = model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    ).logits

                # Get the predicted token IDs for the masked positions
                predicted_token_ids = torch.argmax(logits[0, masked_indices], dim=-1)

                # Create a modified version of the input IDs with predictions
                modified_input_ids = input_ids.clone()
                for i, masked_idx in enumerate(masked_indices):
                    modified_input_ids[0, masked_idx] = predicted_token_ids[i]

                # Decode the augmented text (excluding special tokens)
                augmented_text = tokenizer.decode(modified_input_ids[0].cpu().numpy(), skip_special_tokens=True)

            # Create a new record with the augmented text and other original columns
            new_record = row.to_dict()
            new_record['text_input'] = augmented_text
            augmented_records.append(new_record)

    # Convert the list of augmented records into a new DataFrame
    augmented_train_dataset = pd.DataFrame(augmented_records)

    # 4. Save the augmented dataset to CSV
    try:
        augmented_train_dataset.to_csv(save_path, index=False)
        print(f"✅ Augmented dataset successfully saved to '{save_path}'.")
    except Exception as e:
        print(f"❌ Error saving augmented dataset to '{save_path}': {e}")

    print("✨ Data augmentation complete!")
    return augmented_train_dataset