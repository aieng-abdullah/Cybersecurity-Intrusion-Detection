import os
import pandas as pd
import logging
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from joblib import dump
import datetime 


# Lgger Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()],
)
logger = logging.getLogger()


#  Constants
DATA_PATH = r"C:\Users\teamp\Desktop\Cybersecurity Intrusion Detection\data\preprocess\preprocessed_data.csv"   
MODEL_PATH = r"C:\Users\teamp\Desktop\Cybersecurity Intrusion Detection\src\models\output\IDS.joblib"   
TARGET_COLUMN = "attack_detected"                
LOG_PATH = "logs/train.log"
os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
# Loading Data


def load_data(path: str) -> pd.DataFrame:

    """
     Load preprocessed data from a CSV file.

    """
    if not os.path.exists(path):
        logger.error(f"Data file not found {path}")
        raise FileNotFoundError(f"Data file not found {path}")
    
    logger.info(f"Loding data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded succesfully with shape: {df.shape}")
    return df


# Train Model 

def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train RandomForest using GridSearchCV."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    logger.info("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    return grid_search.best_estimator_

    model.fit(X_train, y_train)



#   MODEL SAVING

def save_model(model, base_dir: str, base_name: str):
    """
    Save the model safely with overwrite warning and versioning.

    """
    save_path = os.path.join(base_dir, base_name)

    if os.path.exists(save_path):
        logger.warning(f"⚠️ Model already exists at {save_path}. Overwriting...")

    # Save main model
    dump(model, save_path)
    logger.info(f" Model saved to: {save_path}")

    # Save a versioned backup with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_name = f"model_{timestamp}.joblib"
    versioned_path = os.path.join(base_dir, versioned_name)
    dump(model, versioned_path)
    logger.info(f" Versioned model backup saved to: {versioned_path}")




 #  MAIN ENTRY POINT
def main():
    """
    Main training pipeline: load data, split, train model, and save.
    """
    logger.info(" IDS model training started...")

    try:
        df = load_data(DATA_PATH)

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

        # Prepare features and labels
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # Train-test split
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train the model
        best_model = train_model(X_train, y_train)

        # Save the trained model
        dump(best_model, MODEL_PATH)
        logger.info(f" Model saved to: {MODEL_PATH}")

    except Exception as e:
        logger.exception(f" Training pipeline failed: {e}")

    logger.info(" IDS model training pipeline completed.")


if __name__ == "__main__":
    main()