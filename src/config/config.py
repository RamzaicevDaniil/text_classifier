from pydantic.settings import BaseSettings


class Settings(BaseSettings):
    raw_data_path: str
    processed_data_path: str
    model_save_path: str
    vectorizer_save_path: str
    api_host: str
    api_port: int

    class Config():
        env_file = "../config.yaml"
        env_file_encoding = 'utf-8'


settings = Settings()
