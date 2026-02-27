# utils.py
import os
import sys
import pickle
import logging

from src.exception import CustomException

def save_object(file_path: str, obj) -> None:
    """
    Save an object as a pickle file.
    
    Parameters:
        file_path (str): Path to save the pickle file.
        obj: Python object to save.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the object
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        
        logging.info(f"Object saved successfully at: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """
    Load a pickled object.
    
    Parameters:
        file_path (str): Path of the pickle file to load.
    
    Returns:
        Loaded Python object.
    """
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict) -> dict:
    """
    Train and evaluate multiple models with hyperparameter tuning.

    Parameters:
        X_train, y_train: Training data
        X_test, y_test: Testing data
        models (dict): Dictionary of model_name: model_instance
        param (dict): Dictionary of model_name: hyperparameter_grid
    
    Returns:
        model_report (dict): Dictionary with test F1 score and trained model for each model
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score
    
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}...")
            grid = GridSearchCV(model, param[model_name], cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            
            model_report[model_name] = {"model": best_model, "test_f1": f1}
            logging.info(f"{model_name} Test F1 Score: {f1:.4f}")
        
        return model_report
    
    except Exception as e:
        raise CustomException(e, sys)