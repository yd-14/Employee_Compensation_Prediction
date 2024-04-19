'''
    Module to ingest data from sql database 
    Train test split
    Save raw data
    Save train and test data as csv
'''

import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


##Step 1: Setup config
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw_data.csv')
    train_data_path = os.path.join('artifacts', 'train_data.csv')
    test_data_path = os.path.join('artifacts', 'test_data.csv')


##Step2: Create a class for data ingestion

class DataIngestion:
    def __init__(self) -> None:
        self.config=DataIngestionConfig()
    
    def data_ingest(self):
        logging.info('Data ingestion started')
        #1. read data from database
        #2. store raw data as csv
        #3. split the raw data into train and test
        #4. store the train and test csv 
        #5. return train and test data paths

        try:
            #Code for reading from db

            df = pd.DataFrame(pd.read_csv('notebooks\data\Employee_Compensation.csv'))

            logging.info('DataFrame read from Database')
            
            #make directory if it does not exist
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)

            #store raw data
            df.to_csv(self.config.raw_data_path, index=False)

            #train test split
            train_data, test_data = train_test_split(df, test_size=0.25, random_state=45)

            #store train data
            train_data.to_csv(self.config.train_data_path, index=False, header=True)

            #store test data
            test_data.to_csv(self.config.test_data_path, index=False, header=True)

            logging.info('Data ingested successfully')

            #return paths to train and test data
            return (
                self.config.train_data_path,
                self.config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        


