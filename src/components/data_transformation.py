import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer  # Creating a pipeline (weather to do one_hot, or standardscaler)
from sklearn.impute import SimpleImputer #If we have some missing values in df we can use this 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacrs','preprocessor.pkl')

class Datatransformation:
    '''
    This function is responsible for data transformation based on different types of data
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps =[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )    
            logging.info("Categorical columns standard scaling completed")
            
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException (e,sys)
    def initiate_data_transformation(self,train_path,test_path):  
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            logging.info("Obtaining preprocessing object")
            prprocessing_obj = self.get_transformer_object()
            
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_data = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_data = test_df[target_column_name]

            logging.info("Applying preprocessing on object on training dataframe and testing data frame")
            input_feature_train_arr = prprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = prprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_ [
                input_feature_train_arr,np.array(target_feature_train_data)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_data)
            ]

            logging.info(f'Saved Preprocessing Object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = prprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
        
 



 