'''
module define classes that can be used to transform raw data into standardized format that can be fed to models for training

input: train data csv path, test data csv path
output: transformed train data, transformed test data, path to preprocessor object

'''

import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_function

from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_object_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    def get_data_transformation_object(self, df):
        '''
        define separate pipeline for numerical and categorical data
        combine them into a preprocessor object and return it
        '''
        try:
            '''
            Step 1: list the categorical and numerical columns
            Step 2: Define numerical and categorical pipelines
            Step 3: Combine using column transformer
            Step 4: return object
            '''

            logging.info('Preprocessor object creation initiated')

            #get list of categorical and numerical column names
            categorical_columns = df.select_dtypes(include = ['object', 'category']).columns.tolist()
            numerical_columns = df.select_dtypes(exclude = ['object', 'category']).columns.tolist()
            

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy = 'median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy = 'most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)), #handle unknown data in test or prediction
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info('Pipeline object created successfully')
            
            return preprocessor

        except Exception as e:
            logging.info('Error in creating data transformation object')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        read saved train and test data
        separate target column
        transform data using preprocessing object
        '''
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')


            target_column_name = 'Salaries'
            drop_columns = [target_column_name]

            #separate target column
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            #get the preprocessing object
            preprocessing_obj = self.get_data_transformation_object(input_feature_test_df)
            #improvement needed
            #instead of passing dataframe pass names of columns separate as numerical and categorical

            ## Transform using preprocessor obj
            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            #add target column to the back of the arr
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #save the preprocessing object
            save_function(file_path=self.config.preprocessor_object_path, obj=preprocessing_obj)
            logging.info('Preprocessor pickle file saved')


            #return the transformed data and preprocessing object
            return (
                train_arr,
                test_arr,
                self.config.preprocessor_object_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)


# train_data_path = 'artifacts/train_data.csv'
# test_data_path = 'artifacts/test_data.csv'
# data_transformation = DataTransformation()
# train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
# print('done')



