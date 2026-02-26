import os
import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")



class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Handle imbalance for XGBoost
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

            
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),

                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "SGD Classifier": SGDClassifier(
                    loss="log_loss",
                    max_iter=1000,
                    random_state=42
                ),

                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    tree_method="hist"
                )
            }


            params = {

                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10]
                },

                "Random Forest": {
                    "n_estimators": [200, 300],
                    "max_depth": [10, 15, 20],
                    "min_samples_split": [5, 10],
                    "min_samples_leaf": [2, 4]
                },

                "SGD Classifier": {
                    "alpha": [0.0001, 0.001, 0.01]
                },


                "XGBoost": {
                    "n_estimators": [300, 500],
                    "learning_rate": [0.01, 0.05],
                    "max_depth": [3, 5],
                    "min_child_weight": [1, 3],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "gamma": [0, 0.1],
                    "reg_alpha": [0, 0.1],
                    "reg_lambda": [1, 5]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            
            best_model_name = None
            best_model = None
            best_f1 = 0

            for model_name, model_info in model_report.items():

                if model_info["test_f1"] > best_f1:
                    best_f1 = model_info["test_f1"]
                    best_model = model_info["model"]
                    best_model_name = model_name

            print("\n==============================================")
            print(f"BEST MODEL: {best_model_name}")
            print(f"BEST TEST F1: {best_f1:.4f}")
            print("==============================================")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Test F1 Score: {best_f1:.4f}")

            return best_f1

        except Exception as e:
            raise CustomException(e, sys)