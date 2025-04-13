import sys
from keras import Sequential
from keras.api.layers import Input, Dense
from keras.api.optimizers import SGD
from keras.api.callbacks import EarlyStopping
from keras.api.optimizers.schedules import ExponentialDecay
from keras.api.initializers import Zeros, RandomNormal
from keras.api.regularizers import l1, l2

from helpers import preprocess_train_data, TRAINED_MODEL

# prepare dataset
dataset_path = sys.argv[1]
print('Loading and preprocessing data...', end=' ')
X_train, Y_train = preprocess_train_data(dataset_path)
print('[DONE]')
# create neural network and add layers
print('Creating and configuring neural network...', end=' ')
nn = Sequential()
nn.add(Input(shape=(X_train.shape[-1],)))
nn.add(Dense(units=32, activation='relu', kernel_initializer=RandomNormal(0, 1), kernel_regularizer=l1(0.01)))
nn.add(Dense(units=1, activation='sigmoid', kernel_initializer=Zeros(), kernel_regularizer=l2(0.001)))
lr_schedule = ExponentialDecay(initial_learning_rate=0.1, decay_steps=1600, decay_rate=0.95, staircase=False)
sgd_optimizer = SGD(learning_rate=lr_schedule)
nn.compile(optimizer=sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall'])
early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, restore_best_weights=True)
print('[DONE]')
# train and save trained neural network
print('Training neural network...', end=' ')
nn.fit(X_train, Y_train, epochs=2000, batch_size=16, shuffle=True, class_weight={0:1.5, 1:1}, callbacks=[early_stop])
print('[DONE]')
print(f'Saving neural network as \'{TRAINED_MODEL}\'...', end=' ')
nn.save(TRAINED_MODEL)
print('[DONE]')