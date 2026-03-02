# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    # No automatic saving now (manual selection later)
    artifacts_dir: str = "artifacts"


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            # Split features and target
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Handle class imbalance for XGBoost
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

            # ----------------------------
            # Define Models
            # ----------------------------
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),

                "Random Forest": RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
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

            # ----------------------------
            # Define Hyperparameters
            # ----------------------------
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

                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
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

            # ----------------------------
            # Train + Log via MLflow
            # ----------------------------
            logging.info("Training models and logging to MLflow...")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            logging.info("All models trained and logged successfully to MLflow.")
            print("\nAll models trained and logged to MLflow successfully.")

            # No automatic selection
            return model_report

        except Exception as e:
            raise CustomException(e, sys)