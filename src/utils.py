import os 
import sys 
import joblib
#import dill 
#import pickle 
from sklearn.metrics import r2_score

from src.exception import FileOperationError
from src.log_config import logging 

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