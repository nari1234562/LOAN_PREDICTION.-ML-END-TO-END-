import sys
import mlflow
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training_pipeline():
    try:
        # ✅ Set experiment ONCE here (important)
        mlflow.set_experiment("Loan_Default_Model_Comparison")

        logging.info("Training Pipeline Started")

        # 1️⃣ DATA INGESTION
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # 2️⃣ DATA VALIDATION
        validation = DataValidation()
        validation_status = validation.validate_all_columns()

        if not validation_status:
            logging.error("Data Validation Failed. Pipeline Stopped.")
            return

        logging.info("Data Validation Passed")

        # 3️⃣ DATA TRANSFORMATION
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )
        )

        logging.info("Data Transformation Completed")

        # 4️⃣ MODEL TRAINING (No best model selection now)
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("Model Training Completed Successfully")
        print("\nTraining pipeline finished successfully.")
        print("All models logged to MLflow.")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()