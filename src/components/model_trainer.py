# Train different models and after that what kind of metricx we are getting from each of the models 
import os
import sys 
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
# from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
        def __init__(self):
              self.model_trainer_config = ModelTrainerConfig()
        
        def initiate_model_trainer(self, train_array,test_array):
              try:
                    logging.info('Splitting training test input data')
                    X_train,y_train,X_test,y_test = (
                          train_array[:,:-1], # remove last columns (independent variable)
                          train_array[:,-1], # (Dependent Variable)
                          test_array[:,:-1], #Independent
                          test_array[:,-1], # Dependent
                    ) 
                    models = {
                                "Linear Regression": LinearRegression(),
                                "K-Neighbors Regressor": KNeighborsRegressor(),
                                "Decision Tree": DecisionTreeRegressor(),
                                "Random Forest Regressor": RandomForestRegressor(),
                                # "XGBRegressor": XGBRegressor(), 
                                "AdaBoost Regressor": AdaBoostRegressor()
                            }
                    model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)

                    best_model_score = max(sorted(model_report.values()))

                    best_model_name = list(model_report.keys())[
                          list(model_report.values()).index(best_model_score)
                    ]

                    best_model = models[best_model_name]

                    if best_model_score <0.6:
                          raise CustomException("No best model found")
                    
                    logging.info("Best found model on both trainingb and test dataset")

                    save_object(
                          file_path = self.model_trainer_config.trained_model_file_path,
                          obj = best_model
                    )

                    predicted = best_model.predict(X_test)

                    r2_square = r2_score(y_test , predicted) 
                    return r2_square
                                           
              except Exception as e:
                    CustomException(e, sys)

