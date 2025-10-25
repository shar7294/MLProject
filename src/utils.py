import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging


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