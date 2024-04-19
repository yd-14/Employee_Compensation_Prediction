'''
Random functions used throughout the project
'''

import os 
import sys 
import pickle 
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging


# takes in path and object and save it as pickle file
def save_function(file_path, obj): 
    #get the path
    dir_path = os.path.dirname(file_path)

    #make sure directory exists
    os.makedirs(dir_path, exist_ok= True)

    #save the file
    with open (file_path, "wb") as file_obj: 
        pickle.dump(obj, file_obj)

def model_performance(X_train, y_train, X_test, y_test, models): 
    '''
    Input: train and test data, model list
    Output: performance report (R2 score) of all the models
    '''
    try: 
        #report dict to be returned
        report = {}
        for i in range(len(models)): #loop through all the model train them and test
            model = list(models.values())[i]
            model.fit(X_train, y_train) # Train models
            y_test_pred = model.predict(X_test) # Test data
            #R2 Score for evaluation
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e: 
        raise CustomException(e,sys)

# Function to load a object from pickle file 
def load_obj(file_path):
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object fuction in utils")
        raise CustomException(e,sys)