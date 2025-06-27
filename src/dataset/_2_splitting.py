import pandas as pd
from sklearn.model_selection import train_test_split


def slit_dataset(dataset: pd.DataFrame):
    """
    split dataset into (70% train, 10% validation, 20% test)
    """

    # Determine stratification
    min_samples_per_class = dataset["assignee_encoded"].value_counts().min()
    stratify_column = None

    if min_samples_per_class < 3:
        print(
            f"Warning: Some classes have fewer than 2 samples (min_samples_per_class={min_samples_per_class}). Stratification for train/test split might not be possible or effective."
        )
    else:
        stratify_column = dataset["assignee_encoded"]
        print("Stratification will be attempted for train/test split.")

    val_train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42, stratify=stratify_column
    )

    stratify_val_train_final = None
    min_samples_per_class = val_train_dataset["assignee_encoded"].value_counts().min()
    if stratify_column is not None:
        val_train_assignee_counts_min = (
            val_train_dataset["assignee_encoded"].value_counts().min()
        )
        if val_train_assignee_counts_min < 2:
            print(
                f"Warning: After first split, some classes in val_train_dataset have fewer than 2 samples{min_samples_per_class}=({min_samples_per_class})"
            )
        else:
            stratify_val_train_final = val_train_dataset["assignee_encoded"]
            print(
                "Stratification for a subsequent train/val split from val_train_dataset appears feasible."
            )

    train_dataset, validation_dataset = train_test_split(
        val_train_dataset,
        test_size=0.125,
        random_state=42,
        stratify=stratify_val_train_final,
        shuffle=True,
    )

    return train_dataset, test_dataset, validation_dataset
