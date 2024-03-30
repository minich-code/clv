import sys 
import os 
from dataclasses import dataclass 

import numpy as np 
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from src.exception import FileOperationError
from src.log_config import logging 

from src.utils import save_object

# Define the dataclass to hold the data transformation configuration. It is a container for holding the transformed data 
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("data_repository", "preprocessor.joblib")

# Define a class for data transformation responsible for performing actual data transformation 
class DataTransformation:
    def __init__(self):
        # Initialize the DataTransformationConfig object
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            categorical_columns = ['State', 'Response', 'Coverage', 'Education', 'Employment Status', 'Gender', 'Location',
                                    'Marital Status', 'Policy Type', 'Policy', 'Renew Offer Type', 'Sales Channel', 'Vehicle Class', 'Vehicle Size']
            numerical_columns = ['Income', 'Monthly Premium Auto', 'Months Since Last Claim', 'Months Since Policy Inception',
                                 'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount']
            
            # numerical_columns = ['Income', 'Months Since Last Claim', 'Months Since Policy Inception', 'Number of Open Complaints',
            #                      'Number of Policies']
            
            # one_hot_columns = ['State', 'Employment Status', 'Location', 'Marital Status', 'Policy Type', 'Sales Channel', 'Vehicle Class']
            
            # label_encode_columns = ['Response', 'Coverage', 'Education', 'Gender', 'Policy', 'Renew Offer Type', 'Vehicle Size']
           
            # outlier_cols = ['Monthly Premium Auto', 'Total Claim Amount']

           # Create a pipeline for numeric columns 
            num_pipeline = Pipeline(
                steps=[
                    # Impute missing values with mean as strategy  
                    ('imputer', SimpleImputer(strategy='mean')),
                    # Scale data using standard scaler
                    ('scaler', StandardScaler())
                ]
            )
            # Create a pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    # Impute missing values with the most frequent value
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # One hot encode the categorical data
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    # Scale the data using StandardScaler
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

                  # Create a column transformer to combine the numeric and categorical pipelines 
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ],
                remainder='passthrough' # specifies how to handle columns that are not explicitly mentioned in the transformers list
            )
            # Log message 
            logging.info("Data preprocessing completed using ColumnTransformer.")

            return preprocessor

        except Exception as e:
            raise FileOperationError(e, sys)

# Initiate data transformation after data ingestion
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load the train and test data 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Log info 
            logging.info("Data loaded successfully.")

            # log information about obtaining the preprocessing object 
            logging.info("Obtaining preprocessing object")

            # Get the preprocessing object 
            preprocessor_obj = self.get_data_transformer_obj()

            # Define the target column name
            target_col = "Customer Lifetime Value"
            
            # Drop the target column from the input features in both training and testing sets 
            train_input_features = train_df.drop(columns = [target_col], axis=1)
            train_target_feature = train_df[target_col]

            test_input_features = test_df.drop(columns = [target_col], axis=1)
            test_target_feature = test_df[target_col]

            # Log the application of the preprocessing object on training and testing DataFrames 
            logging.info("Applying preprocessing object on training and testing DataFrames")

            # Fit the preprocessing object to the training and transform the training and testing set 
            train_input_features_array = preprocessor_obj.fit_transform(train_input_features)
            test_input_features_array = preprocessor_obj.transform(test_input_features) 

            # Log info about successful data transformation 
            logging.info("Data transformation completed successfully.")

            # Concatenate the transformed input features with target features for both training and testing sets 
            training_array = np.c_[train_input_features_array, np.array(train_target_feature)]
            testing_array = np.c_[test_input_features_array, np.array(test_target_feature)]

            # Log the saving of the preprocessing object 
            logging.info("Saving preprocessing object")

            # Save the preprocessing object 
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )


            # Return the transformed train and test arrays and the preprocessing file path
            return (
                training_array,
                testing_array,
                self.data_transformation_config.preprocessor_obj_file_path

            )

        except Exception as e:  
            raise FileOperationError(e, sys)

      