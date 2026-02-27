# src/pipeline/predict_pipeline.py
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class CustomData:
    """
    Class to structure input data for prediction using actual dataset columns.
    """
    def __init__(
        self,
        age,
        gender,
        education_level,
        annual_income,
        employment_experience_years,
        home_ownership_status,
        loan_amount,
        loan_purpose,
        interest_rate,
        loan_to_income_ratio,
        credit_history_length_years,
        credit_score,
        prior_default_flag
    ):
        self.age = age
        self.gender = gender
        self.education_level = education_level
        self.annual_income = annual_income
        self.employment_experience_years = employment_experience_years
        self.home_ownership_status = home_ownership_status
        self.loan_amount = loan_amount
        self.loan_purpose = loan_purpose
        self.interest_rate = interest_rate
        self.loan_to_income_ratio = loan_to_income_ratio
        self.credit_history_length_years = credit_history_length_years
        self.credit_score = credit_score
        self.prior_default_flag = prior_default_flag

    def get_data_as_dataframe(self):
        """
        Convert input features into a pandas DataFrame with actual column names.
        """
        try:
            data_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "education_level": [self.education_level],
                "annual_income": [self.annual_income],
                "employment_experience_years": [self.employment_experience_years],
                "home_ownership_status": [self.home_ownership_status],
                "loan_amount": [self.loan_amount],
                "loan_purpose": [self.loan_purpose],
                "interest_rate": [self.interest_rate],
                "loan_to_income_ratio": [self.loan_to_income_ratio],
                "credit_history_length_years": [self.credit_history_length_years],
                "credit_score": [self.credit_score],
                "prior_default_flag": [self.prior_default_flag],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    """
    Class to load model + preprocessor and make predictions.
    """
    def __init__(self):
        try:
            self.model_path = "artifacts/model.pkl"
            self.preprocessor_path = "artifacts/preprocessor.pkl"
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame):
        """
        Make predictions on a DataFrame.
        Returns both predictions and probability of positive class (1).
        """
        try:
            # Transform input
            input_arr = self.preprocessor.transform(df)

            # Predict
            predictions = self.model.predict(input_arr)

            # Probability of positive class
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(input_arr)[:, 1]
            else:
                probabilities = [None] * len(predictions)

            return predictions, probabilities

        except Exception as e:
            raise CustomException(e, sys)