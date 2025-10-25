import os 
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pickle
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessort_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def detect_feature_types(self, train_data_path:str, target_column:str):
        """
        Automatatically detects numerical and categorical features in the dataset.
        Args:
            train_data_path (str): Path to the training data CSV file.
            target_column (str): Name of the target column.
    
        """
        logging.info("Detecting feature types.")
        try:
            df = pd.read_csv(train_data_path)
            numerical_features = df.select_dtypes(exclude='object').columns.tolist()
            categorical_features = df.select_dtypes(include='object').columns.tolist()

            # Remove target column from features list

            if target_column in numerical_features:
                numerical_features.remove(target_column)
            if target_column in categorical_features:
                categorical_features.remove(target_column)
            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")
            return numerical_features, categorical_features
        except Exception as e:
            logging.error("Error in detecting feature types.")
            raise CustomException(e, sys)
    
    def get_data_transformer_object(self, numerical_features, categorical_features):
        """
        Creates and returns a preprocessing pipeline that handles:
        - Missing values
        - Scaling numeric features
        - One-hot encoding categorical features
        Args:
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
        """
        logging.info("Creating preprocessing pipeline started.")

        try:
            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical pipeline
            çategorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_pipeline, numerical_features),
                    ('cat_pipeline', çategorical_pipeline, categorical_features)
                ]
            )

            logging.info("Preprocessing pipeline created successfully.")
            return preprocessor
        
        except Exception as e:
            logging.error("Error in creating preprocessing pipeline.")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path:str, test_data_path:str, target_column:str):
        """
        Applies transformations to train and test datasets:
        Fits transformer on training data
        Transforms both train and test sets
        Saves preprocessor to artifacts
        Returns transformed arrays
        """

        logging.info("Data transformation initiated.")
        try:
            # Load datasets
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Detect feature types
            numerical_features, categorical_features = self.detect_feature_types(train_data_path, target_column)

            # Get preprocessing object
            preprocessor = self.get_data_transformer_object(numerical_features, categorical_features)

            # Separate features and target
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            # Fit and transform training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Combine transformed features with target
            train_array = np.c_[X_train_transformed, y_train.to_numpy()]
            test_array = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save preprocessor object

            save_object(
                file_path=self.transformation_config.preprocessort_obj_file_path,
                obj=preprocessor
            ) 

            return train_array, test_array, self.transformation_config.preprocessort_obj_file_path
            
        except Exception as e:
            logging.error("Error in data transformation process.")
            raise CustomException(e, sys)
               

        



    