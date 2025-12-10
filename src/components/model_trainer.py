import os
import sys
from dataclasses import dataclass
from src.models import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras 
import tensorflow as tf
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            print(X_train.shape)

            # Create models
            models = {
                "CB_model": CB_model(),
                "CF_model": CF_model(X_train, y_train)
            }

            loss = tf.losses.MeanAbsoluteError()
            model_report = {}

            # Train and evaluate each model separately
            for name, model in models.items():
                print(f"Training {name}...")
                # Create a fresh optimizer for each model
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
                
                # Compile model
                model.compile(optimizer=optimizer, loss=loss)
                
                # Fit model
                model.fit(
                    [X_train[:, 0], X_train[:, 1]], y_train,
                    validation_split=0.2,
                    epochs=5,
                    batch_size=32,
                    verbose=1
                )

                # Predict and calculate score
                preds = model.predict([X_test[:, 0], X_test[:, 1]])
                score = mean_squared_error(y_test, preds)
                model_report[name] = score
                print(f"{name} MSE: {score}")

            # Select best model
            best_model_name = min(model_report, key=model_report.get)
            best_model = models[best_model_name]
            print(f"Best model: {best_model_name} with MSE {model_report[best_model_name]}")

            # Save model
            best_model.save("artifacts/model_data.h5")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report[best_model_name]

        except Exception as e:
            raise CustomException(e, sys)
