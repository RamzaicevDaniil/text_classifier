from pydantic_settings import BaseSettings
from pydantic import Field
import os


class Settings(BaseSettings):
    raw_data: str = Field(env="RAW_DATA")
    dataset: str = Field(env="DATASET")
    model_save_name: str = Field(env="MODEL_SAVE_NAME")
    host: str = Field(env="HOST")
    port: str = Field(env="PORT")
    model_load_path: str = Field(env="MODEL_LOAD_PATH")
    ml_model_config: str = Field(env="ML_MODEL_CONFIG")

    class Config:
        env_file = os.path.join('../.env')
        env_file_encoding = 'utf-8'


settings = Settings()
