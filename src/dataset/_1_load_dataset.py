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
            f"Summary = {dataset['Summary'].iloc[i]} | Description = {dataset['Description'].iloc[i]}"
        )
    dataset["text_input"] = temp


def jdt_perpocess():
    pass


def sun_perpocess():
    pass


def load_dataset(dataset_type: DatasetType, dataset_rpath: str):
    dataset_path = Path(os.getcwd()) / f"{dataset_rpath}"
    try:
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logging.error(
            f"Error: dataset {dataset_type.value} not found at '{dataset_path}' or in the current directory. Please place it correctly."
        )
        exit()

    gcc_perpocess(dataset)

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
