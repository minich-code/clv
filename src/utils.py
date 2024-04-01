import os 
import sys 
import joblib
#import dill 
#import pickle 
from sklearn.metrics import r2_score

from src.exception import FileOperationError
from src.log_config import logging 

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Define a function to save an object in a file 
def save_object(file_path, obj):
    try:
        # Extract the directory path from the given file path. 
        
        # Get the directory path of the file 
        dir_path = os.path.dirname(file_path)
        # if the directory path does not exist, create it 
        os.makedirs(dir_path, exist_ok = True)

        # save the object in the file 
        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_obj)

    # Handle exception 
    except Exception as e:
        raise FileOperationError(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, model):
    try:
        #logging.info("Fitting the model")
        # Fit the model
        model.fit(X_train, y_train)

        #logging.info("Making predictions")
        # Make predictions on X_train
        #y_train_pred = model.predict(X_train)
        # Make predictions on X_test
        y_test_pred = model.predict(X_test)

        #logging.info("Calculating R-squared scores")
        # Calculate R-squared scores
        #train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        return test_model_score
    
    except Exception as e:
        raise FileOperationError(e, sys)
    


def load_data(file_path):
    """Loads the dataset from the specified file path."""
    return pd.read_csv(file_path)


def encode_columns(df, categorical_columns, label_columns):
    """Encodes categorical and label columns in the DataFrame."""
    # One-hot encoding categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop=None)

    for column in categorical_columns:
        one_hot_encoded = encoder.fit_transform(df[[column]])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df, one_hot_df], axis=1)
        df = df.drop([column], axis=1)

    # Label encoding label columns
    label_encoders = {}

    for col in label_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    return df


def remove_outliers_iqr(df, column_name, threshold=1.5):
    """Removes outliers from the specified column using the IQR method."""
    # Calculate quartiles for the specified column
    q25 = df[column_name].quantile(0.25)
    q75 = df[column_name].quantile(0.75)
    iqr = q75 - q25

    # Calculate the lower and upper bounds for outliers using the IQR method
    lower_bound = q25 - threshold * iqr
    upper_bound = q75 + threshold * iqr

    # Filter the DataFrame to remove rows with values outside the lower and upper bounds
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    # Print the number of outliers removed and the size of the filtered DataFrame
    outliers_count = len(df) - len(df_filtered)
    print(f"Number of outliers removed for '{column_name}' using IQR method: {outliers_count}")
    print(f"Size of filtered DataFrame for '{column_name}' using IQR method: {len(df_filtered)}")

    return df_filtered


def save_data(df, file_path):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file_path, index=True)
