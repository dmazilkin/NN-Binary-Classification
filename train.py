import sys
from keras import Sequential
from keras.api.layers import Input, Dense
from keras.api.optimizers import SGD
from keras.api.callbacks import EarlyStopping

from helpers import preprocess_data, TRAINED_MODEL

# prepare dataset
dataset_path = sys.argv[1]
print('Loading and preprocessing data...', end=' ')
X_train, Y_train = preprocess_data(dataset_path, encode_classes=True)
print('[DONE]')
# create neural network and add layers
print('Creating and configuring neural network...', end=' ')
neural_network = Sequential()
neural_network.add(Input(shape=(X_train.shape[-1],)))
neural_network.add(Dense(units=1, activation='sigmoid'))
sgd_optimizer = SGD(learning_rate=0.1)
neural_network.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, restore_best_weights=True)
print('[DONE]')
# train and save trained neural network
print('Training neural network...', end=' ')
neural_network.fit(X_train, Y_train, epochs=2000, batch_size=64, shuffle=True, callbacks=[early_stop], verbose=0)
print('[DONE]')
print(f'Saving neural network as \'{TRAINED_MODEL}\'...', end=' ')
neural_network.save(TRAINED_MODEL)
print('[DONE]')