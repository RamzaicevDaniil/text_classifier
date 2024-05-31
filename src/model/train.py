import os
import logging
import yaml
import datetime
import pandas as pd
from model.text_classifier import TextClassifier
from sklearn.metrics import classification_report
from config.config import settings


def create_experiment_folder(base_path="experiments"):
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

    experiment_path = create_experiment_folder()

    # Чтение YAML файла
    with open(settings.ml_model_config, 'r') as file:
        config = yaml.safe_load(file)

    save_config(experiment_path, config)

    # Настройка логирования
    log_file = os.path.join(experiment_path, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logger = logging.getLogger()

    # Загрузка данных
    data = pd.read_csv(f"{settings.dataset}/train.csv")

    # Создание и обучение модели
    classifier = TextClassifier(max_features=config['model']['max_features'])
    classifier.train(data['text'], data['category'])

    # Оценка модели
    predictions = classifier.predict(data['text'])
    report = classification_report(data['category'], predictions)
    logger.info("Classification Report:\n" + report)

    # Сохранение модели
    # os.makedirs(os.path.dirname(settings.model_save_name), exist_ok=True)
    classifier.save(f"{os.path.join(experiment_path, settings.model_save_name)}")

    # Сохранение логов и конфигурации


    save_log(experiment_path, "Experiment completed successfully.\n" + report)
    logger.info(f"Experiment logs and configuration saved to {experiment_path}")


if __name__ == "__main__":
    train_pipeline()