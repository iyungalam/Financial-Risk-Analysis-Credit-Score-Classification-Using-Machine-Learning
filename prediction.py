from pathlib import Path
import joblib

# Menentukan lokasi folder project
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Load model dan encoder
model = joblib.load(MODEL_DIR / "gboost_model.joblib")
result_target = joblib.load(MODEL_DIR / "encoder_target.joblib")


def prediction(data):
    """Making prediction

    Args:
        data (Pandas DataFrame): Dataframe that contains all the data
        needed to make prediction.

    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]

    return final_result
