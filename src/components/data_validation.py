import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    schema_status_file: str = os.path.join("artifacts", "validation_status.txt")
    # All 14 columns from your dataset
    required_columns: list = (
        "age", 
        "gender", 
        "education_level", 
        "annual_income", 
        "employment_experience_years",
        "home_ownership_status", 
        "loan_amount", 
        "loan_purpose", 
        "interest_rate", 
        "loan_to_income_ratio",
        "credit_history_length_years", 
        "credit_score", 
        "prior_default_flag", 
        "loan_status"
    )

class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            # Load raw data from ingestion
            data = pd.read_csv(os.path.join("artifacts", "data.csv"))
            all_cols = list(data.columns)

            for col in self.validation_config.required_columns:
                if col not in all_cols:
                    validation_status = False
                    logging.error(f"Validation Error: Column {col} is missing!")
            
            with open(self.validation_config.schema_status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)