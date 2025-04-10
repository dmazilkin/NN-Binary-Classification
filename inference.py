import sys
from keras.api.saving import load_model
from keras import Sequential

from helpers import preprocess_data, TRAINED_MODEL

# prepare dataset
dataset_path = sys.argv[1]
print('Loading and preprocessing data...', end=' ')
X, Y_target = preprocess_data(dataset_path, encode_classes=False)
print('[DONE]')
# loading neural network
print('Loading trained neural network...', end=' ')
neural_network: Sequential = load_model(TRAINED_MODEL)
print('[DONE]')
# apply neural network to the dataset
print('Predicting...', end=' ')
Y_predict = neural_network.predict(X, verbose=0)
print('[DONE]')
print(20 * '-')
print('class'.ljust(5) + '|'.center(10) + 'class_expected'.ljust(5))
Y_predict = (Y_predict > 0.5).astype(int)
Y_predict_labels = get_labels(Y_predict)
if Y_target is not None:
    for label_predict, label_target in zip(Y_predict_labels, Y_predict_labels):
        print(label_predict.ljust(5) + '|'.center(10) + label_target.ljust(5))
else:
    for label_predict in Y_predict_labels:
        print(label_predict.ljust(5))