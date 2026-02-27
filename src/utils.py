import os
import sys
import pickle
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save a Python object to the given file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from the given file path using pickle.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        return obj

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train, tune (via GridSearchCV), and evaluate multiple models.
    Returns a report dictionary containing metrics and the best model.
    """
    try:
        report = {}
        threshold = 0.6   

        for model_name, model in models.items():

            print("\n====")
            print(f"MODEL: {model_name}")
            print("======")

            model_params = param.get(model_name, {})

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=model_params,
                cv=3,
                scoring="f1",
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            print("\nBest Parameters:")
            print(grid_search.best_params_)

            
            if hasattr(best_model, "predict_proba"):
                y_train_prob = best_model.predict_proba(X_train)[:, 1]
                y_test_prob = best_model.predict_proba(X_test)[:, 1]

                y_train_pred = (y_train_prob >= threshold).astype(int)
                y_test_pred = (y_test_prob >= threshold).astype(int)

                train_auc = roc_auc_score(y_train, y_train_prob)
                test_auc = roc_auc_score(y_test, y_test_prob)
            else:
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                train_auc = None
                test_auc = None

        
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)

            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            cm = confusion_matrix(y_test, y_test_pred)

    
            print("\nTrain Metrics:")
            print(f"Precision: {train_precision:.4f}")
            print(f"Recall:    {train_recall:.4f}")
            print(f"F1 Score:  {train_f1:.4f}")
            if train_auc is not None:
                print(f"ROC-AUC:   {train_auc:.4f}")

            print("\nTest Metrics:")
            print(f"Precision: {test_precision:.4f}")
            print(f"Recall:    {test_recall:.4f}")
            print(f"F1 Score:  {test_f1:.4f}")
            if test_auc is not None:
                print(f"ROC-AUC:   {test_auc:.4f}")

            print("\nConfusion Matrix (Test):")
            print(cm)

        
            report[model_name] = {
                "model": best_model,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "train_auc": train_auc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_auc": test_auc,
                "confusion_matrix": cm,
                "best_params": grid_search.best_params_
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)