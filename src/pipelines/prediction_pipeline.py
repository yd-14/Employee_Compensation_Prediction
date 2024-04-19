'''
 Define module to predict using model already trained
'''

import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        '''
        Create a dataframe of data recieved from other sources
        '''
        def __init__(self,
                     Organization_Group_Code:int,
                     Job_Family_Code:str,
                     Job_Code:str,
                     Year:int,
                     Department_Code:str,
                     Union_Code:float,
                     Overtime_Amount:str,
                     Retirement_Amount:str,
                     Health_and_Dental_Amount:str,
                     Other_Benefits_Amount:str): 
            
            # NO COMMAS HERE NO COMMAS
             self.Organization_Group_Code = Organization_Group_Code
             self.Job_Family_Code = Job_Family_Code
             self.Job_Code = Job_Code
             self.Year = Year
             self.Department_Code = Department_Code
             self.Union_Code = Union_Code
             self.Overtime_Amount = Overtime_Amount
             self.Retirement_Amount = Retirement_Amount
             self.Health_and_Dental_Amount = Health_and_Dental_Amount
             self.Other_Benefits_Amount = Other_Benefits_Amount
        
        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = { 
                        'Organization Group Code': [self.Organization_Group_Code],
                        'Job Family Code': [self.Job_Family_Code],
                        'Job Code': [self.Job_Code],
                        'Year': [self.Year],
                        'Department Code': [self.Department_Code],
                        'Union Code': [self.Union_Code],
                        'Overtime Amount': [self.Overtime_Amount],
                        'Retirement Amount': [self.Retirement_Amount],
                        'Health and Dental Amount': [self.Health_and_Dental_Amount], # REMOVE UNDERSCORE
                        'Other Benefits Amount': [self.Other_Benefits_Amount]
                  }
                  df = pd.DataFrame(custom_data_input_dict)
                #   print(df.columns)
                  logging.info("Dataframe created")
                  return df
             except Exception as e:
                  logging.info("Error occured in get_data_as_dataframe function in prediction_pipeline")
                  raise CustomException(e,sys)
             

# data = CustomData(
#             Organization_Group_Code = 3,
#             Job_Family_Code = "1400",
#             Job_Code = "1404",
#             Year = 2019,
#             Department_Code = "HSA",
#             Union_Code = 790.0,
#             Overtime_Amount = "few overtime",
#             Retirement_Amount = "few retirement",
#             Health_and_Dental_Amount = "few Health/Dental",
#             Other_Benefits_Amount = "few Other Benefits"
#         )

# new_data = data.get_data_as_dataframe()
# predict_pipeline = PredictPipeline()
# pred = predict_pipeline.predict(new_data)
# print('ok')