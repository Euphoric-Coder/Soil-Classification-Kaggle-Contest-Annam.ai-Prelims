import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tqdm import tqdm


def load_metadata(base_dir):
    train_csv = os.path.join(base_dir, "train_labels.csv")
    test_csv = os.path.join(base_dir, "test_ids.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df = train_df[train_df["label"] == 1]
    train_df["image_path"] = train_df["image_id"].apply(
        lambda x: os.path.join(base_dir, "train", x)
    )
    test_df["image_path"] = test_df["image_id"].apply(
        lambda x: os.path.join(base_dir, "test", x)
    )
    return train_df, test_df


def load_images(df, image_size=(128, 128)):
    images = []
    for path in tqdm(df["image_path"], desc="Loading images"):
        img = load_img(path, target_size=image_size)
        img = img_to_array(img) / 255.0
        images.append(img)
    return np.array(images)
