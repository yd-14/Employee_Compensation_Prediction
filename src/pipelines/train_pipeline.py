'''
Training pipeline for the regression model
Connects DataIngestion, DataTransformation and Model Trainer
'''

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__": 
    #get data and store as csv
    obj = DataIngestion()
    train_data_path, test_data_path = obj.data_ingest()

    #load data from csv and transform it
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    #train the model using the transformed data
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)