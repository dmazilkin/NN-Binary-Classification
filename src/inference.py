import sys
from keras.api.saving import load_model
from keras import Sequential

from helpers import preprocess_test_data, decode_labels, TRAINED_MODEL

# prepare dataset
dataset_path = sys.argv[1]
print('Loading and preprocessing data...', end=' ')
X, Y_target = preprocess_test_data(dataset_path)
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
Y_predict = (Y_predict > 0.5).astype(int)
Y_predict_labels = decode_labels(Y_predict)
if Y_target is not None:
    print('class' + ' ' + 'class_expected')
    for y_predict, y_target in zip(Y_predict_labels, Y_target):
        print(f'{y_predict}' + ' ' + f'{y_target}')
else:
    print('class'.ljust(5))
    for y_predict in Y_predict:
        print(f'{y_predict}'.ljust(5))