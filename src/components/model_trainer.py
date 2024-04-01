import os
import sys 
from dataclasses import dataclass 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.exception import FileOperationError
from src.log_config import logging 

from src.utils import save_object #, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("data_repository", "model.pkl")
    #trained_model_file_path = os.path.join("data_repository", "model_pkl")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, training_array, testing_array):
        try:
            logging.info("Splitting training and and testing input data")

            # Split train and test arrays into features and target 
            X_train, y_train, X_test, y_test = (
                training_array[:, :-1],
                training_array[:, -1],
                testing_array[:, :-1],
                testing_array[:, -1],
            )

            logging.info("Creating model")
            # Your model
            model = RandomForestRegressor()
            # Fit model
            model.fit(X_train, y_train)

            # Predict using model 
            predict = model.predict(X_test)

            # Calculate the R-squared score 
            model_score = r2_score(y_test, predict)
            logging.info(f"Model score: {model_score}")

            # # Check if model score is less than 0.7
            # if model_score < 0.7:
            #     raise FileOperationError("Model Score is less than 0.7", sys)

            
            logging.info("Saving model")

            save_object(
                file_path = self.config.trained_model_file_path, 
                obj=model
            )

            return model_score
        
        except Exception as e:
            raise FileOperationError(e, sys)
