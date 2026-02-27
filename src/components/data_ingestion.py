import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

# Components
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer  


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
            df = pd.read_csv(r"notebook\loan_cleaned_dataset_.xls")
            logging.info("Dataset read successfully")

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
                'loan_status': 'loan_status'
            })

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train-Test split initiated")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["loan_status"]
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


