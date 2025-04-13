# Neural network usage for solving binary classification problem

This repo contents the two following sections: 
## 1. Visualization of training process and dataset analysis
Dataset analysis, preprocessing, metrics and other analyses are shown in Jupyter notebook **binary_classification.ipynb**.
 ## 2. Source files for neural network deployment
Three Python files can be found in *src/* folder:
1. **train.py** - contains logic for neural network training,
2. **inference.py** - contains logic for predicting classes after training is completed,
3. **helpers.py** - contains some useful functions and variables, which are used by train.py and inference.py.

NOTE: after train.py is completed, the *output/* folder is created. It contains three files:
1. neural_network.keras (*TRAINED_MODEL* variable in helpers.py) - contains trained neural network,
2. standard_scaler.json (*STANDARD_SCALER* variable in helpers.py) - contains Standard Scaler configs,
3. encoder.json (*ENCODER* variable in helpers.py) - contains Encoder configs.

# Usage
All requirements need for running scripts can be found in *requirements.txt*.
## Train the neural network
To train the neural network run:
```console
python src/train.py data/p1_train.csv
```
## Predict classes with trained neural network
To predict classes with trained neural network run:
```console
python src/inference.py data/p1_test_student.csv
```