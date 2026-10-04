import os
import sys
import logging
from abc import ABC, abstractmethod
from src.exception import CustomException
import src.logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


class Logger(ABC):
    @abstractmethod
    def log(self, level, message):
        """Write a log entry at the given logging level."""
        raise NotImplementedError

    @abstractmethod
    def info(self, message):
        """Write an informational log entry."""
        raise NotImplementedError

    @abstractmethod
    def warning(self, message):
        """Write a warning log entry."""
        raise NotImplementedError

    @abstractmethod
    def error(self, message):
        """Write an error log entry."""
        raise NotImplementedError

    @abstractmethod
    def exception(self, message):
        """Write an exception log entry with traceback."""
        raise NotImplementedError


class FileLogger(Logger):
    def __init__(self, file_path="artifacts/app.log"):
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        self.logger = logging.getLogger("ml_project_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def log(self, level, message):
        self.logger.log(level, message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def exception(self, message):
        self.logger.exception(message)


logger = FileLogger()
if not hasattr(src.logger, "logging"):
    src.logger.logging = logger


class PandasDataAdapter(ABC):
    @abstractmethod
    def read_csv(self, file_path, **kwargs):
        """Load a CSV file into a DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def train_test_split(self, df, test_size=0.2, random_state=None, **kwargs):
        """Split data into train and test sets."""
        raise NotImplementedError

    @abstractmethod
    def save_csv(self, dataframe, file_path, **kwargs):
        """Persist a DataFrame to disk as CSV."""
        raise NotImplementedError


class PandasDataAdapterImpl(PandasDataAdapter):
    def read_csv(self, file_path, **kwargs):
        return pd.read_csv(file_path, **kwargs)

    def train_test_split(self, df, test_size=0.2, random_state=None, **kwargs):
        return train_test_split(df, test_size=test_size, random_state=random_state, **kwargs)

    def save_csv(self, dataframe, file_path, **kwargs):
        dataframe.to_csv(file_path, **kwargs)
        return file_path


pandas_data_adapter = PandasDataAdapterImpl()

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        src.logger.logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook\\data\\stud.csv")
            src.logger.logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            src.logger.logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            src.logger.logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



