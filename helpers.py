import pandas as pd
from typing import Union
from sklearn.preprocessing import StandardScaler, LabelEncoder

TRAINED_MODEL = 'neural_network_weights.keras'

def preprocess_data(file: str, encode_classes: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Preprocess data from dataset.

    :param file: path to CSV file with dataset
    :param encode_classes: True for encoding classes, False - otherwise

    :return: X train, Y train
    '''

    # read CSV file and split data into X and Y dataframes
    X, Y = None, None
    df = pd.read_csv(file)
    if 'class' in df.columns:
        X, Y = df.drop(['class'], axis=1), df['class']
    else:
        X = df
    # preprocess X data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # preprocess Y data if defined
    if Y is not None and encode_classes:
        encoder = LabelEncoder()
        encoder.fit(Y)
        Y = encoder.transform(Y)
    return X, Y