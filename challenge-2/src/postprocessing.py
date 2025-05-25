import os
import pandas as pd


def save_submission(test_df, predictions, output_path=None):
    # ✅ If output_path is not provided, save in challenge-2/data/submission.csv
    if output_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
        output_path = os.path.join(base_dir, "submission.csv")

    # ✅ Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": predictions})
    submission.to_csv(output_path, index=False)
    print(
        f"\n✅ Submission saved at {output_path}. Predicted {sum(predictions)} images as soil."
    )
