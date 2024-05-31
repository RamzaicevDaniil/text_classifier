import pandas as pd
import os
from config.config import settings
import zipfile

def preprocess_data(input_path: str, output_path: str):
    # Загружаем данные
    # data = pd.read_csv(input_path)
    #
    # # Проверяем данные
    # assert 'label' in data.columns and 'text' in data.columns, "Данные должны содержать колонки 'label' и 'text'"
    #
    # # Сохраняем предобработанные данные
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # data.to_csv(output_path, index=False)


    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)

if __name__ == "__main__":
    preprocess_data(settings.raw_data, settings.dataset)
