import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import json


def load_train_data(train_csv_path, train_folder):
    train_df = pd.read_csv(train_csv_path)
    train_df["path"] = train_df["image_id"].apply(
        lambda x: os.path.join(train_folder, x)
    )
    train_df = train_df[train_df["path"].apply(os.path.exists)]
    return train_df


def encode_labels(train_df, label_column="soil_type"):
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df[label_column])
    return train_df, le


def compute_class_weights(labels):
    """Compute balanced class weights for imbalanced datasets."""
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


def save_label_encoder(le, save_path):
    with open(save_path, "w") as f:
        json.dump(le.classes_.tolist(), f)


def load_label_encoder(load_path):
    with open(load_path) as f:
        classes = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    return le
