import json
import re
import ast
from pathlib import Path
from pprint import pprint
from typing import Dict, Any, List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go

gcc_result = {
    "mlp": {
        "glove_mean": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert_mean": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert_cls": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "cnn": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-dot": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-concat": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-general": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-multi-head": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-dot": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-concat": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-general": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-multi-head": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-dot-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-concat-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-general-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-multi-head-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-dot-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-concat-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-general-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-multi-head-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    }
}

jdt_result = {
    "mlp": {
        "glove_mean": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert_mean": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert_cls": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "cnn": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-dot": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-concat": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-general": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-multi-head": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-dot": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-concat": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-general": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-multi-head": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-dot-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-concat-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-luong-general-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "lstm-attention-multi-head-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-dot-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-concat-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-luong-general-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    },
    "bi-lstm-attention-multi-head-residual": {
        "glove": {"top-1":0, "top-5":0, "top-10":0, "mrr":0},
        "bert": {"top-1":0, "top-5":0, "top-10":0, "mrr":0}
    }
}


METRIC_KEYS = [
    'test_top_5_acc',
    'test_top_10_acc',
    'test_acc_top1_final',
    'test_mrr_final',
]

MAP_METRIC_KEYS = {
    'test_acc_top1_final': "top-1",
    'test_top_5_acc': "top-5",
    'test_top_10_acc': "top-10",
    'test_mrr_final': "mrr",
}

def preprocess_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the nested dictionary into a tidy pandas DataFrame,
    standardizing embedding keys to prevent palette errors.
    """
    records = []
    for model, embeddings in data.items():
        for embed_key, metrics in embeddings.items():
            # --- THIS IS THE KEY FIX ---
            # Standardize various embedding names into one of two keys
            # that will match the palette dictionary.
            if 'glove' in embed_key.lower():
                embedding_type = 'GloVe'
            elif 'bert' in embed_key.lower():
                embedding_type = 'BERT'
            else:
                embedding_type = 'Unknown'

            for metric, value in metrics.items():
                score = value / 100.0 if metric == 'top-1' and value > 1 else value
                records.append({
                    'model': model,
                    'embedding': embedding_type, # Use the standardized name
                    'metric': metric,
                    'score': score
                })
    return pd.DataFrame(records)

def find_metrics_in_text(text: str) -> Optional[Dict[str, float]]:
    key_pos = -1
    for key in METRIC_KEYS:
        for quote in ["'", '"']:
            pos = text.find(f"{quote}{key}{quote}")
            if pos != -1:
                key_pos = pos
                break
        if key_pos != -1:
            break

    if key_pos == -1:
        return None

    start_pos = text.rfind('{', 0, key_pos)
    if start_pos == -1:
        return None

    brace_level = 0
    end_pos = -1
    substring = text[start_pos:]
    for i, char in enumerate(substring):
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
        
        if brace_level == 0:
            end_pos = start_pos + i
            break
    
    if end_pos == -1:
        return None

    dict_str = text[start_pos : end_pos + 1]

    dict_str = re.sub(r'\bnan\b', "'nan'", dict_str, flags=re.IGNORECASE)
    dict_str = re.sub(r'\btrue\b', 'True', dict_str, flags=re.IGNORECASE)
    dict_str = re.sub(r'\bfalse\b', 'False', dict_str, flags=re.IGNORECASE)
    dict_str = re.sub(r'\bnull\b', 'None', dict_str, flags=re.IGNORECASE)

    try:
        data = ast.literal_eval(dict_str)
        if isinstance(data, dict):
            metrics = {}
            for key in METRIC_KEYS:
                if key in data:
                    value = data[key]
                    if isinstance(value, str) and value.lower() == 'nan':
                        metrics[key] = float('nan')
                    else:
                        metrics[key] = value

            if all(key in metrics and isinstance(metrics.get(key), (int, float)) for key in METRIC_KEYS):
                return metrics
    except (ValueError, SyntaxError, MemoryError) as e:
        print(f"      [!] Could not parse dictionary string. Error: {e}")
        return None
    
    return None


def extract_metrics_from_notebook(notebook_path: Path) -> List[Dict[str, float]]:
    print(f"\n--- Processing Notebook: {notebook_path.name} ---")
    found_metrics = []
    
    try:
        with notebook_path.open('r', encoding='utf-8') as f:
            notebook_content = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [!] Error reading or parsing notebook file: {e}")
        return []

    for i, cell in enumerate(notebook_content.get('cells', [])):
        if cell.get('cell_type') != 'code' or not cell.get('outputs'):
            continue

        for output in cell.get('outputs', []):
            text_content = ""
            if 'text' in output:
                text_content = "".join(output['text'])
            elif 'data' in output and 'text/plain' in output['data']:
                text_content = "".join(output['data']['text/plain'])

            if text_content:
                metrics = find_metrics_in_text(text_content)
                if metrics:
                    print(f"  [+] Metrics found in cell {i+1}")
                    found_metrics.append(metrics)
                    break 
                    
    if not found_metrics:
        print("  [-] No target metrics found in this notebook.")
        
    return found_metrics


def process_directory(directory_path: str) -> Dict[str, List[Dict[str, float]]]:
    path = Path(directory_path)
    if not path.is_dir():
        print(f"Error: Directory not found at '{directory_path}'")
        return {}

    all_results = {}
    notebook_files = sorted(list(path.rglob('*.ipynb')))
    
    if not notebook_files:
        print(f"No Jupyter Notebooks found in '{directory_path}' or its subdirectories.")
        return {}

    print(f"Found {len(notebook_files)} notebook(s). Starting processing...")

    for notebook_path in notebook_files:
        results = extract_metrics_from_notebook(notebook_path)
        if results:
            all_results[notebook_path.name] = results
            
    return all_results


def preprocess_data(data: dict) -> pd.DataFrame:
    """
    Transforms the nested JSON data into a flat pandas DataFrame.

    This function iterates through the nested dictionary, extracting the model name,
    embedding type, metric, and value to create a structured DataFrame suitable
    for plotting and analysis.

    Args:
        data: A dictionary containing the model performance data.

    Returns:
        A pandas DataFrame with columns: 'model', 'embedding', 'metric', 'value'.
    """
    records = []
    for model, embeddings in data.items():
        for embedding, metrics in embeddings.items():
            for metric, value in metrics.items():
                records.append({
                    'model': model,
                    'embedding': embedding,
                    'metric': metric,
                    'value': value
                })
    df = pd.DataFrame(records)
    # Create a combined model-embedding name for unique labeling on the plot
    df['model_embedding'] = df['model'] + ' (' + df['embedding'] + ')'
    return df

def plot_and_save_metric(df: pd.DataFrame, metric_name: str, output_dir: str, dataset_name: str):
    """
    Generates, saves, and displays an interactive bar chart for a specific metric.

    This function filters the DataFrame for a given metric, sorts the models
    by performance, and creates a bar chart using Plotly Express. The chart is
    customized, saved to a file, and then displayed.

    Args:
        df: The preprocessed pandas DataFrame.
        metric_name: The name of the metric to plot (e.g., 'top-1', 'mrr').
        output_dir: The directory where the plot image will be saved.
    """
    metric_df = df[df['metric'] == metric_name].sort_values('value', ascending=False)

    fig = px.bar(
        metric_df,
        x='model_embedding',
        y='value',
        color='model',  # Color bars by the base model name for grouping
        text='value',
        title=f'Model Performance Comparison for {metric_name.upper()}',
        labels={'model_embedding': 'Model and Embedding', 'value': f'{metric_name.upper()} Score'},
        height=800
    )

    # --- Advanced Customization ---
    fig.update_layout(
        xaxis_title="Model (Embedding)",
        yaxis_title=f"Score ({'Percentage' if metric_name == 'top-1' else 'Ratio'})",
        xaxis={'categoryorder':'total descending'}, # Keep the sorted order
        legend_title="Base Model",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="RebeccaPurple"
        ),
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template='plotly_white'
    )

    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    
    fig.update_xaxes(tickangle=45)

    # --- Save the figure ---
    # Define the full file path
    file_path = os.path.join(output_dir, f'{dataset_name}_{metric_name}_performance.png')
    
    # Save the figure to the specified path
    # Using a higher resolution for better quality in the saved file
    print(f"Saving plot to {file_path}...")
    fig.write_image(file_path, width=1600, height=900)
    print("Save complete.")

    # --- Display the figure ---
    fig.show()


def main(json_data, dataset_name):
    """
    Main function to run the data processing, plotting, and saving.
    """
    # Define the output directory and create it if it doesn't exist
    output_dir = 'final_metric_diagrams'

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    # 1. Preprocess the data into a usable format
    df = preprocess_data(json_data)

    # 2. Identify all unique metrics in the data
    metrics_to_plot = df['metric'].unique()

    # 3. Generate and save a plot for each metric
    for metric in metrics_to_plot:
        print(f"\n--- Generating plot for {metric} ---")
        plot_and_save_metric(df, metric, output_dir, dataset_name)
        print(f"--- Finished plot for {metric} ---")



if __name__ == '__main__':

    target_directory = '.' 
    final_results = process_directory(target_directory)

    print("\n\n" + "="*50)
    print("          METRIC EXTRACTION SUMMARY")
    print("="*50 + "\n")

    if not final_results:
        print("could not extract the data.\nexitting")
        exit(1)
    print("transfering result to data structure for draw diagrams ... ")
    for key, value in final_results.items():
        embedding_model, dl_model = key.split('-')[0], "-".join(key.split('-')[1:]).split(".")[0]
        gcc_result[dl_model][embedding_model], value[0]
        jdt_result[dl_model][embedding_model], value[1]
        
        for k, v in value[0].items():
            gcc_result[dl_model][embedding_model][MAP_METRIC_KEYS[k]] = v
        
        for k, v in value[1].items():
            jdt_result[dl_model][embedding_model][MAP_METRIC_KEYS[k]] = v
    
    print("draw diagrams and save in ./output folder")
    print(gcc_result)
    print(jdt_result)

    for jdata, dname in [(gcc_result, "gcc"), (jdt_result, "jdt")]:
        main(jdata, dname)