import os 
import sys 
from dataclasses import dataclass 

import pandas as pd 
from sklearn.model_selection import train_test_split 

from src.exception import FileOperationError
from src.log_config import logging

# Import after data transformation 
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

# Import after model trainer 
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


# Define a dataclass that will hold data ingestion configuration 
# This is a container for holding configuration data such as training, testing and raw data 
@dataclass
class DataIngestionConfig:
    # The path to the raw, train and tests data file path 
    # We indicate that data will be stored in folder called data_repository 
    # Raw data path
    raw_data_path: str = os.path.join("data_repository", "raw.csv")
    # Train data path
    train_data_path: str = os.path.join("data_repository", "train.csv")
    # Test data path
    test_data_path: str = os.path.join("data_repository", "test.csv")


# Define a dataclass for data input. This will perform the actual data importation 
class DataIngestion:
# Define a constructor to initiate the data ingestion class
    def __init__(self, config: DataIngestionConfig):
        # Store the configuration data 
        self.config = config

        # Method to initiate data importation process
    def initiate_data_ingestion(self):
        # Log message to indicate start of data ingestion process 
        logging.info("Starting data ingestion process")
        try:
            # Read the raw data
            df = pd.read_csv(r"E:\MLproject\customerclv\notebook\Insurance Customer Lifetime Value.csv")
            # Log message to indicate successful reading of dataframe 
            logging.info("Successfully read the raw data")

            # Create a directory for train data if it does not exist 
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            # Save the dataframe to the raw data path 
            df.to_csv(self.config.raw_data_path, index = False, header=True)
            logging.info("Successfully saved the raw data to the data repository folder")

            # Log message to indicate end of data importation process and saving raw data 
            logging.info("Exited the data ingestion process")

            # Log message to commence splitting to training and testing
            logging.info("Starting data splitting process")
            # Split the data into train and test sets 
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            # Save the train and test sets to csv files to train data path 
            train_set.to_csv(self.config.train_data_path, index = False, header=True)
            test_set.to_csv(self.config.test_data_path, index = False, header=True)

            # Logging info
            logging.info("Successfully saved the train and test sets to the data repository folder")

            # Return the paths to the train and test data 
            return (self.config.train_data_path, self.config.test_data_path)
        
        
        except Exception as e:
            raise FileOperationError(e, sys)
        


# if __name__ == "__main__":
#      config = DataIngestionConfig()
#      obj=DataIngestion(config)
#      obj.initiate_data_ingestion()

# if __name__=="__main__":
#     config = DataIngestionConfig()
#     obj = DataIngestion(config) # creates an instance of the DataIngestion class, which is responsible for data ingestion operations.
#     train_data, test_data =  obj.initiate_data_ingestion() # This line calls the initiate_data_ingestion() method of the DataIngestion.

#     data_transformation = DataTransformation() # Creates an instance of the DataTransformation class, responsible for data transformation operations.
#     data_transformation.initiate_data_transformation(train_data, test_data) # This method initiates the data transformation process
        
# After model trainer 
if __name__=="__main__":
    config = DataIngestionConfig()
    obj = DataIngestion(config) 
    train_data, test_data =  obj.initiate_data_ingestion() 

    data_transformation = DataTransformation() 
    training_array, testing_array, _ = data_transformation.initiate_data_transformation(train_data, test_data) 

    config = ModelTrainerConfig()
    model_trainer = ModelTrainer(config)
    print(model_trainer.initiate_model_trainer(training_array, testing_array))