import os
import gzip
import shutil
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple

MODEL_URL = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
COMPRESSED_FILENAME = "GoogleNews-vectors-negative300.bin.gz"
MODEL_FILENAME = "GoogleNews-vectors-negative300.bin"
VECTOR_DIMENSION = 300

def get_word2vec_vectors(
    save_dir: str = "word2vec_model"
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Checks for a local Word2Vec model, downloads it if not present,
    and returns a vector map and an embedding for unknown words.
    
    This version parses the .bin file manually and does NOT require 'gensim'.

    This function is designed to be efficient and user-friendly. It shows a
    progress bar during download and handles file operations safely.

    The "unknown" embedding is calculated as the mean of all vectors in the
    vocabulary, which is a common and effective strategy.
    """
    print("--- Word2Vec Vectorization Initialized (Gensim-Free) ---")
    save_dir_path = Path(save_dir)
    model_path = save_dir_path / MODEL_FILENAME
    compressed_path = save_dir_path / COMPRESSED_FILENAME

    # --- Step 1: Ensure directory exists ---
    save_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Model directory set to: {save_dir_path.resolve()}")

    # --- Step 2: Check for model, download and decompress if necessary ---
    if not model_path.is_file():
        print(f"Model file not found at {model_path}")
        if not compressed_path.is_file():
            print(f"Compressed model not found. Downloading from {MODEL_URL}...")
            try:
                # Streaming download with a progress bar (tqdm)
                with requests.get(MODEL_URL, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kilobyte
                    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {COMPRESSED_FILENAME}")
                    with open(compressed_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            progress_bar.update(len(chunk))
                            f.write(chunk)
                    progress_bar.close()

                if total_size != 0 and progress_bar.n != total_size:
                    raise IOError("ERROR: Download incomplete.")
                print("Download complete.")

            except requests.exceptions.RequestException as e:
                print(f"Error downloading the file: {e}")
                if compressed_path.exists():
                    os.remove(compressed_path)
                return {}, np.zeros(VECTOR_DIMENSION, dtype=np.float32)

        print(f"Decompressing {compressed_path}...")
        try:
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Decompression complete.")
        except Exception as e:
            print(f"An error occurred during decompression: {e}")
            if model_path.exists():
                os.remove(model_path)
            return {}, np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    else:
        print("Found existing Word2Vec model file. Skipping download.")

    # --- Step 3: Load model and create vector map MANUALLY ---
    print("Loading Word2Vec model into memory. This may take a moment...")
    vectors_map = {}
    all_vectors = []
    try:
        with open(model_path, "rb") as f:
            # Read header
            header = f.readline().decode('utf-8')
            vocab_size, vec_dim = map(int, header.split())
            print(f"Model details: {vocab_size} words, {vec_dim} dimensions.")
            
            if vec_dim != VECTOR_DIMENSION:
                raise ValueError(f"Model dimension mismatch! Expected {VECTOR_DIMENSION}, got {vec_dim}")

            binary_len = np.dtype(np.float32).itemsize * vec_dim
            
            # Read vocabulary and vectors
            for i in tqdm(range(vocab_size), desc="Parsing model file"):
                # Read word
                word_bytes = bytearray()
                while True:
                    char_byte = f.read(1)
                    if char_byte == b' ':
                        break
                    if char_byte != b'\n': # Skip newline chars at the end of some lines
                        word_bytes.extend(char_byte)
                word = word_bytes.decode('utf-8', errors='ignore')
                
                # Read vector
                vector_bytes = f.read(binary_len)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                vectors_map[word] = vector
                all_vectors.append(vector)

        print("Model loaded and parsed successfully.")
        
        # --- Step 4: Create the 'unknown' embedding ---
        # Calculate the mean of all vectors to create a generic 'unk' vector
        print("Calculating 'unk_embedding'...")
        unk_embedding = np.mean(all_vectors, axis=0)
        print("Calculated 'unk_embedding' as the mean of all vectors.")
        
        print("--- Vectorization Complete ---")
        return vectors_map, unk_embedding

    except Exception as e:
        print(f"Failed to load or parse the model. Error: {e}")
        # If the file is corrupted, remove it so a fresh download is triggered next time
        if model_path.exists():
            os.remove(model_path)
            print(f"Removed potentially corrupted model file: {model_path}")
        return {}, np.zeros(VECTOR_DIMENSION, dtype=np.float32)