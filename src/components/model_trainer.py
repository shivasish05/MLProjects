# Train different models and after that what kind of metricx we are getting from each of the models 
import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
        def __init__(self):
              self.model_trainer_config = ModelTrainer()
        
        def initiate_model_trainer(self, train_array,test_array, preprocessor_path):
              try:
                    loggin
              except:
                    pass

