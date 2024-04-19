'''
Module to train various models and save the best one
'''

import os
import sys
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.ensemble import RandomForestRegressor

from src.utils import save_function 
from src.utils import model_performance 

#define path to save model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array): 
        '''
        separate target and features
        list the models used in training
        fit and check performance of each model
        save the best model based on r2 score
        '''
        try: 
            logging.info('Seggregating the dependent and independent variables')

            #Seperate the last column i.e., the target column
            X_train, y_train, X_test, y_test = (train_array[:, :-1], 
                                                train_array[:,-1], 
                                                test_array[:, :-1], 
                                                test_array[:,-1])
            
            #Regression models
            models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(), 
                'Lasso':Lasso(), 
                'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            }
            model_report: dict = model_performance(X_train, y_train, X_test, y_test, models)

            #display all the model results
            print(model_report)
            print('-'*100)
            logging.info(f'Model Report: {model_report}')

            # Best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]            
            best_model = models[best_model_name]

            #display the best model selected
            print(f'The best model is {best_model_name}, with R2 Score: {best_model_score}')
            print('-'*100)
            logging.info(f'The best model is {best_model_name}, with R2 Score: {best_model_score}')

            #save the best model
            save_function(file_path = self.model_trainer_config.trained_model_file_path, obj = best_model)
            logging.info('Best model saved')


        except Exception as e: 
            logging.info('Error occured during model training')
            raise CustomException(e,sys)