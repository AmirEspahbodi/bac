import os
import logging
from pathlib import Path
import pandas as pd
from ._dataset_types import DatasetType
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def gcc_perpocess(dataset: pd.DataFrame):
    dataset["Summary"] = dataset["Summary"].fillna("").astype(str)
    dataset["Description"] = dataset["Description"].fillna("").astype(str)
    temp = []
    for i in range(len(dataset["Summary"])):
        temp.append(
            f"{dataset['Summary'].iloc[i]} {dataset['Description'].iloc[i]}"
        )
    dataset["text_input"] = temp


def jdt_perpocess(dataset: pd.DataFrame):
    dataset["text_input"] = dataset["Description"].fillna("").astype(str)


def sun_perpocess(dataset: pd.DataFrame):
    dataset["text_input"] = dataset["Description"].fillna("").astype(str)


def load_dataset(dataset_type: DatasetType):
    current_dir = Path(os.getcwd())
    try:
        match dataset_type:
            case DatasetType.GCC:
                dataset_rpath="./datasets/gcc_data.csv"
                dataset_processor = gcc_perpocess
            case DatasetType.JDT:
                dataset_rpath="./datasets/jdt_data.csv"
                dataset_processor = jdt_perpocess
            case DatasetType.SUN:
                dataset_rpath="./datasets/sun_firefox.csv"
                dataset_processor = sun_perpocess
            case DatasetType.CUSTOM:
                raise NotImplementedError("not suported yet")
            case _:
                raise RuntimeError("wrong dataset") 
        dataset = pd.read_csv(current_dir / f"{dataset_rpath}")
    except FileNotFoundError:
        logging.error(
            f"Error: dataset {dataset_type} not found at '{dataset_rpath}' or in the current directory. Please place it correctly."
        )
        exit()

    try:
        dataset_processor(dataset)
    except Exception as e:
        logging.error(
            f"Error: error while processing dataset e: {e}."
        )       

    # Drop rows with missing Assignee
    dataset.dropna(subset=["Assignee"], inplace=True)
    dataset["Assignee"] = dataset["Assignee"].astype(
        str
    )  # Ensure assignee names are strings
    print(f"Shape after dropping NA assignees: {dataset.shape}")

    # Label Encoding for Assignee
    assignee_encoder = LabelEncoder()
    dataset["assignee_encoded"] = assignee_encoder.fit_transform(dataset["Assignee"])
    NUM_ACTUAL_CLS = len(assignee_encoder.classes_)
    print(f"Number of unique assignees (classes): {NUM_ACTUAL_CLS}")

    return dataset, NUM_ACTUAL_CLS
