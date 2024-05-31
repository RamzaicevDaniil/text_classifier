import requests
import os
from config import settings

def download_data(url: str, output_path: str):
    response = requests.get(url)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(response.content)

if __name__ == "__main__":
    download_data('https://example.com/train_data.csv', settings.raw_data_path)