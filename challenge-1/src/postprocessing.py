import os
import pandas as pd
import numpy as np
from collections import Counter


def predict_and_prepare_submission(
    model, test_flow, test_df, test_csv_path, le, output_path
):
    """Predict on test data and prepare the submission file."""
    preds = model.predict(test_flow)
    pred_labels = le.inverse_transform(np.argmax(preds, axis=1))
    print("Prediction class distribution:", Counter(pred_labels))

    full_test_df = pd.read_csv(test_csv_path)
    filtered_test_df = test_df[
        test_df["image_id"].isin(test_flow.filenames)
    ].reset_index(drop=True)

    predicted_df = pd.DataFrame(
        {"image_id": filtered_test_df["image_id"], "soil_type": pred_labels}
    )
    submission = full_test_df.merge(predicted_df, on="image_id", how="left")
    submission["soil_type"] = submission["soil_type"].fillna("Alluvial soil")

    submission.to_csv(output_path, index=False)
    print(f"Submission file saved at: {output_path}")
    return submission


def get_image_data_generator(rescale=1.0 / 255, **kwargs):
    """Get an ImageDataGenerator with specified augmentations."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(rescale=rescale, **kwargs)
