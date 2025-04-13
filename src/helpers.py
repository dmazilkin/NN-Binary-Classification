import pandas as pd
import json
from typing import Any
from sklearn.preprocessing import StandardScaler, LabelEncoder

TRAINED_MODEL = 'output/neural_network.keras'
STANDARD_SCALER = 'output/standard_scaler.json'
ENCODER = 'output/encoder.json'

def preprocess_train_data(file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Preprocess train data from dataset.

    :param file: path to CSV file with dataset
    :param encode_classes: True for encoding classes, False - otherwise

    :return: X train, Y train
    '''

    # read CSV file and split data into X and Y dataframes
    df = pd.read_csv(file)
    X, Y = df.drop(['class'], axis=1), df['class']
    # preprocess X data
    scaler = StandardScaler()
    scaler.fit(X)
    with open(STANDARD_SCALER, 'w') as file:
        json.dump({'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}, file, indent=4)
    X = scaler.transform(X)
    # preprocess Y data if defined
    encoder = LabelEncoder()
    encoder.fit(Y)
    with open(ENCODER, 'w') as file:
        json.dump({'classes': encoder.classes_.tolist()}, file, indent=4)
    Y = encoder.transform(Y)
    return X, Y

def preprocess_test_data(file: str):
    '''
    Preprocess test data from dataset.

    :param file: path to CSV file with dataset

    :return: X test, Y test
    '''

    df = pd.read_csv(file)
    X, Y = df.drop(['class'], axis=1), df['class']
    scaler = StandardScaler()
    with open(STANDARD_SCALER, 'r') as file:
        scaler_config = json.load(file)
    scaler.mean_, scaler.scale_ = scaler_config['mean'], scaler_config['scale']
    X = scaler.transform(X)
    return X, Y

def decode_labels(Y: Any) -> pd.Series:
    encoder = LabelEncoder()
    with open(ENCODER, 'r') as file:
        encoder_config = json.load(file)
    encoder.classes_ = pd.Series(encoder_config['classes'], name='class', dtype='object')
    return pd.Series(encoder.inverse_transform(Y))
