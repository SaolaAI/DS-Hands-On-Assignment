import os
import dotenv
dotenv.load_dotenv()
import json
import requests

HOST = os.getenv("HOST")
PORT = os.getenv("PORT")


def get_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def fit(train_path):
    train_data = get_data(train_path)
    response = requests.post(f'http://{HOST}:{PORT}/fit', json=train_data)
    if response.status_code == 200:
        print("Data successfully sent to /fit endpoint")
    else:
        print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")


def predict(val_path):
    val_data = get_data(val_path)
    result = []
    response = requests.post(f'http://{HOST}:{PORT}/predict', json=val_data)
    return result

def health_check():
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    response = requests.get(f'http://{host}:{port}/health')
    if response.status_code == 200:
        print("Health check passed")
    else:
        print(f"Health check failed. Status code: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    train_path = "..." # Path to your training data
    val_path = "..." # Path to your validation data
    fit(train_path)
    result = predict(val_path)
    print(result)
