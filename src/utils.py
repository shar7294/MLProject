import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path: str, obj) -> None:
    """
    Saves a Python object (like a preprocessor, model, etc.) to the given file path using pickle.

    Args:
        file_path (str): Path where the object should be saved.
        obj: The Python object to be saved.
    """

    try:
        # Create directory if it doesn’t exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params):
    """
    Evaluates multiple machine learning models and returns their performance scores.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        models (dict): A dictionary where keys are model names and values are model instances.
        Param_distributions (dict): A dictionary where keys are model names and values are 
        hyperparameter distributions for tuning.
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            
            # Train the model
            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)

            # mode.fit(X_train, y_train)

            # Predict on train data 
            y_train_pred = best_model.predict(X_train)

            # Predicut on test data
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
            
def load_object(file_path: str):
    """
    Loads a Python object from the given file path using pickle.

    Args:
        file_path (str): Path from where the object should be loaded.

    Returns:
        The loaded Python object.
    """

    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
