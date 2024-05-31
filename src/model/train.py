import os
import pandas as pd
from src.models.text_classifier import TextClassifier
from sklearn.metrics import classification_report
import logging
import yaml
import datetime
from config.settings import settings


def create_experiment_folder(base_path="src/experiments"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_path = os.path.join(base_path, f"experiment_{timestamp}")
    os.makedirs(experiment_path, exist_ok=True)
    return experiment_path


def save_config(experiment_path, config):
    config_path = os.path.join(experiment_path, "config.yaml")
    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def save_log(experiment_path, log_content):
    log_path = os.path.join(experiment_path, "log.txt")
    with open(log_path, 'w') as file:
        file.write(log_content)

def train_pipeline():
    # Создание папки эксперимента
    experiment_path = create_experiment_folder()

    # Сохранение конфигурации
    config = {
        "data": {
            "processed_path": settings.dataset
        },
        "model": {
            "max_features": 1000
        }
    }
    save_config(experiment_path, config)

    # Настройка логирования
    log_file = os.path.join(experiment_path, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger()

    # Загрузка данных
    data = pd.read_csv(settings.processed_data_path)

    # Создание и обучение модели
    classifier = TextClassifier()
    classifier.train(data['text'], data['label'])

    # Оценка модели
    predictions = classifier.predict(data['text'])
    report = classification_report(data['label'], predictions)
    logger.info("Classification Report:\n" + report)

    # Сохранение модели
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    classifier.save(settings.model_path)

    # Сохранение логов и конфигурации
    save_log(experiment_path, "Experiment completed successfully.\n" + report)
    logger.info(f"Experiment logs and configuration saved to {experiment_path}")


if __name__ == "__main__":
    train_pipeline()