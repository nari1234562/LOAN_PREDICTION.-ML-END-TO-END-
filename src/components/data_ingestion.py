import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# 2. Validation Connection
from src.components.data_validation import DataValidation

# 3.data Transformation Connection
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # ✅ FIX: Using read_csv because the file is text-based despite the .xls extension
            df = pd.read_csv(r"notebook\loan_cleaned_dataset_.xls")
            logging.info("Dataset read successfully as dataframe")
            df = df.rename(columns={
                'person_age': 'age',
                'person_gender': 'gender',
                'person_education': 'education_level',
                'person_income': 'annual_income',
                'person_emp_exp': 'employment_experience_years',
                'person_home_ownership': 'home_ownership_status',
                'loan_amnt': 'loan_amount',
                'loan_intent': 'loan_purpose',
                'loan_int_rate': 'interest_rate',
                'loan_percent_income': 'loan_to_income_ratio',
                'cb_person_cred_hist_length': 'credit_history_length_years',
                'credit_score': 'credit_score',
                'previous_loan_defaults_on_file': 'prior_default_flag',
                'loan_status':'loan_status'
            })
            logging.info("Columns renamed successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # 1. DATA INGESTION
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # 2. DATA VALIDATION
        from src.components.data_validation import DataValidation
        validation = DataValidation()
        validation_status = validation.validate_all_columns()
        
        if validation_status:
            logging.info("Data Validation Passed. Proceeding to Transformation...")
            
            # 3. DATA TRANSFORMATION
            from src.components.data_transformation import DataTransformation
            data_transformation = DataTransformation()
            # This returns the combined numpy arrays (X and y together)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

            # 4. MODEL TRAINING
        
        else:
            logging.error("Data Validation Failed! Check artifacts/validation_status.txt")
            print("Pipeline stopped: Data format is incorrect.")

    except Exception as e:
        from src.exception import CustomException
        import sys
        raise CustomException(e, sys)