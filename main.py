"""Equity Research
Scott Westvold
Machine Learning Algorithm, Trading Algorithm, and Testing Module"""
################################################################################
# Module Description
################################################################################
#
# This module provides a projected price for the security at the end of the quarter .
# It will be assumed this projected price is the securities mean return as the goal of the neural
# network is to provide sophisticated equity reasearch which will provide an accurate projection.
# This module can be summarized in 3 parts. First there is a Keras Neural Network which has the purposed
# of predicting whether or not the S & P 500 will go up over a user input time.
# The next is a trading algorithm which implements a threshold trading strategy based
# off the predictions of the Neural Network. THe third is a testing module which is built to
# determine the performance of the algorithm using several metrics including testing accuracy, F-Score,
# % of trades positive, and fund performance.
#
################################################################################
# Packages used by this Module
################################################################################
import keras
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import csv
from matplotlib import pyplot as py
from keras.models import Model
from keras import metrics
from keras.layers import Input, Embedding, LSTM, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Nadam
from keras.layers import Dense, LSTM
from keras import regularizers
import dataprocessing
from keras.models import Sequential
import time
from keras.constraints import min_max_norm
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from scipy import stats

################################################################################
# Functions Used by this module
################################################################################
"""
    `NN_price_trainer(training_data, lookbacks, data_shape)`

This function represents the Neural Network and the compilation and training of the Neural Network.
This function takes 3 inputs. `training_data` is a set of `sequence` objects created by the `Keras`
timeseries generator. `lookbacks` is the number of previous trading days associated with each input
`training_data`. `data_shape` is the number of features describing each trading day.

This function then creates a Recurrent Neural Network. Each Layer will be described in line below.
It will compile and fit the model weights to the training data.

This function returns the trained Neural Network which can be accessed later for testing.
"""
def NN_price_trainer(training_data, lookbacks, data_shape):
    r1 = regularizers.l2(0.00001)                                               # Regularization for the network weights as an attempt to bridge the gap between testing and training accuracies
    r2 = regularizers.l1(0.00001)
    dr = .025                                                                   # dropout rate for the dropout layers in order to ignore data noise and try to increase testing accuracy

    price_model = Sequential()                                                  # Sequential Neural Network
                                                                                # BatchNormalization is included but commented out as it is inconclusive whether it improves performance or not but does increase required training time (more epochs would be required)
                                                                                # Left in for potential future work
    #price_model.add(keras.layers.BatchNormalization(input_shape = (lookbacks, 151), axis = 1, momentum = 0.9, epsilon = 0.001, center = True, scale = True))
    price_model.add(LSTM(250, input_shape = (lookbacks, data_shape),  bias_initializer = 'random_uniform', return_sequences = True)) # First Long Short Term Memory Network
    price_model.add(keras.layers.Dropout(dr))
    price_model.add(keras.layers.AlphaDropout(dr, noise_shape = None, seed = None)) # A style of dropout layer specialized for regularizing very noisy data

    price_model.add(LSTM(250, bias_initializer = 'random_uniform', return_sequences = True))
    price_model.add(keras.layers.MaxPooling1D(pool_size = 2, strides = None, padding = 'valid', data_format = 'channels_last')) # Max Pooling to reduce data size for more feasible run time. Also increases focus on network inputs from the previous layer with the largest value and thus hopefuly ignores small moves in features.

    price_model.add(LSTM(250, bias_initializer = 'random_uniform', return_sequences = False))
    price_model.add(keras.layers.Dropout(dr))

    price_model.add(Dense(250, activation = 'linear',  kernel_regularizer = r1, activity_regularizer = r2))
    price_model.add(keras.layers.Dropout(dr))

    price_model.add(Dense(150, activation = 'tanh',  kernel_regularizer = r1, activity_regularizer = r2))  # Tanh activation allows for complex functions as oppsoed to large linear funcitons. In addition tanh provides potential outputs from -1 to 1 allowing for previous layer weights to contribute negative values to the output
    price_model.add(keras.layers.Dropout(dr))

    price_model.add(Dense(70, activation = 'linear',  kernel_regularizer = r1, activity_regularizer = r2))
    price_model.add(Dense(1, activation = 'sigmoid',  kernel_regularizer = r1, activity_regularizer = r2))    # Activation Layer. Sigmoid Activation.

    adg = Adam(amsgrad = False, lr = .0005)                                     # Different styles of optimizers which can be used to speed or slow (but add precision) training
    ndg = Nadam()
    rms2 = RMSprop()
    sgd = SGD()

    call = keras.callbacks.EarlyStopping(monitor = 'binary_crossentropy', min_delta = 0, patience = 40, verbose = 0, mode = 'min', baseline = None, restore_best_weights = True)    # Ways to alter learning process and learning rate in order to improve precision and reduce wasted processing time
    call2 = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.25, patience = 3, verbose = 0, mode = 'min', min_delta = 0.0, cooldown = 0, min_lr = 0)

    price_model.compile(loss = 'binary_crossentropy', optimizer = adg,  metrics = ['binary_accuracy'])
    history = price_model.fit_generator(training_data, epochs = 60, steps_per_epoch = 200, callbacks = [call2], verbose = 2).history

    return price_model                                                          # Return the trained Neural Netowork model to access later for testing.


"""
    NN_tester(model, sequences)

This function is the testing module for a trained Neural Network. This function takes 2 inputs.
`model` is the trained Neural Network that will be used to make predicitons and calculate testing accuracy.
"""
def NN_tester(model, sequences):
        predictions = model.predict_generator(sequences, verbose = 0)
        score = model.evaluate_generator(sequences, steps=None, verbose = 0)
        # print(predictions)                                                    # Uncomment to see predictions and ensure they are a varied and somewhat even distribution
        print("\nAccuracy:", score[1], "Loss:", score[0])
        return predictions, score[1]

"""
    NN_predictor(model, sequences)

This function is the predicting module for a trained Neural Network. This function takes 2 inputs.
`model` is the trained Neural Network that will be used to make predicitons.
"""
def NN_predictor(model, sequences):
        predictions = model.predict_generator(sequences, verbose = 0)
        return predictions

"""
    `run_reward_data(training_examples, testing_examples, offset, lookbacks, projection_length, sentiment, smart_feature_extraction)`

This function loads in the data by calling dataprocessing.py and splits it into training and testing data sets.
This function then generates time series from these data arrays to be fed to the recurrent Neural Network.
This function returns the training data, testing data, data indicies, copied S & P 500 returns matrix `values_2`, the testing labels, and the number of features per trading day `data_shape`.
"""
def run_reward_data(training_examples, testing_examples, offset, lookbacks, projection_length, sentiment, smart_feature_extraction):
    test_ind1 = training_examples                                               # Index into the loaded data array representing the end of the training data
    test_ind2 = training_examples + offset + testing_examples                   # Index into the loaded data array representing the end of the testing data
    data_num = training_examples + testing_examples + offset + 1 + projection_length + 100  # The amount of data to load

    features, labels, values = dataprocessing.run_program(data_num, lookbacks, projection_length, sentiment, smart_feature_extraction)  # Call the dataloading module

    training_features = features[:training_examples, :]
    training_labels = labels[:training_examples, :]
    test_ind1 = training_examples + offset                                      # Update test_ind1 to represent the first index in the testing data
    testing_features = features[test_ind1:test_ind2, :]
    testing_labels = labels[test_ind1:test_ind2, :]

    values_2 = values[test_ind1:test_ind2, :]                                   # Alter NumPy Array Format
    data_shape = np.shape(training_features)[1]                                 # Get the number of features per training day

    training_sequences = TimeseriesGenerator(training_features, training_labels, length = lookbacks, sampling_rate = 1, stride = 1,  batch_size = 1)    # Create the timeseries of "lookbacks" trailing days for training and testing
    test_sequences =  TimeseriesGenerator(testing_features, testing_labels, length = lookbacks, sampling_rate = 1, stride = 1, batch_size = 1)

    return training_sequences, test_sequences, test_ind1, test_ind2, values_2, testing_labels, data_shape

"""
    run_reward_program(training_sequences, testing_sequences, test_ind1, test_ind2, values, labels, lookbacks, projection_length, data_shape)

This function takes the data from the `run_reward_data` function and then calls the Neural Network trainer, tester, and trader in order to run the entire simulation.
It returns 4 performance metrics fund performance, percent of trades positive, f-score, and testing accuracy.
"""
def run_reward_program(training_sequences, test_sequences, test_ind1, test_ind2, values, labels, lookbacks, projection_length, data_shape):
    NN_price_model = NN_price_trainer(training_sequences, lookbacks, data_shape)
    predictions, testing_accuracy = NN_tester(NN_price_model, test_sequences)
    test_ind1 = test_ind1
    test_ind2 = test_ind2 + projection_length
    testing_values = values

    performance, pos_trades, fscore = trade(predictions, testing_values, projection_length, labels)
    return performance, pos_trades, fscore, testing_accuracy, predictions

"""
    `trade(predictions, return_labels, projection_length, testing_labels)`

This function is the trading algorithm module. It employs a threshold trading strategy by only trading if the
prediciton for the probability of an S & P 500 increase is above the 50th percentile of the predictions.
It returns 3 performance metrics of fund performance, f score, and percent of trades which were positive.
"""
def trade(predictions, return_labels, projection_length, testing_labels):
    trades = 1
    pos = 0
    neg = 0
    tot = 0
    pos_ac = 0
    false_pos = 0
    fund_value = 1
    with open('Fund_Performance.csv', mode='w') as test_file2:
        performance_writer = csv.writer(test_file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, np.shape(predictions)[0]):
            ret = float(return_labels[i])                                       # Single Trade return
            if predictions[i] >= np.percentile(predictions, 50):                # 50th Percentile Threshold
                    trades = trades + 1
                    fund_value = fund_value + fund_value/projection_length * (ret)
                    if ret > 0:
                        pos = pos + 1
            else:
                if ret < 0:
                    neg = neg + 1
                else:
                    false_pos = false_pos + 1
            tot = tot + 1
            performance_writer.writerow([fund_value])
    for i in range(0, np.shape(testing_labels)[0]):
        if testing_labels[i, :] == 1:
            pos_ac = pos_ac + 1
    # print("TOTAL POSITIVE TESTING DAYS: ", pos_ac)                            # Other Metrics Which Could be Printed to reveal information about the simulation
    # print("Number of Trades: ", trades)
    # print("Number of Trades with Positive Return: ", pos)
    print("Percent Positive: ", pos/trades)
    # print("Accuracy: ", (pos+neg)/tot)
    print("Fund Performace: ", fund_value)
    if trades > 0:
        trades = trades - 1                                                     # One additional trade was added at the beginning to account for a divide by 0 bug
    precision = pos/trades
    recall = pos/(pos + false_pos)
    fscore = 2 * precision * recall/ (precision + recall)
    return fund_value, pos/trades, fscore

"""
    `run_trials(training_examples, testing_examples, offset, lookbacks, projection_length, trials, sentiment, smart_feature_extraction)`

This function allows for data to only be loaded once and then multiple trials of training and trading to be run in order to get average performances.
THis function returns the averages over the trals of the four performance metrics performance, percent trades positive, f score, and test accuracy.
"""
def run_trials(training_examples, testing_examples, offset, lookbacks, projection_length, trials, sentiment, smart_feature_extraction):
    i = 0
    acc = 0
    tot_per = 0
    tot_pos = 0
    tot_fscore = 0
    tot_accuracy = 0
    training_sequences, test_sequences, test_ind1, test_ind2, values_2, testing_labels, data_shape = run_reward_data(training_examples, testing_examples, offset, lookbacks, projection_length, sentiment, smart_feature_extraction)

    while i < trials:
        pos, per, fscore, acc, predictions = run_reward_program(training_sequences, test_sequences, test_ind1, test_ind2, values_2, testing_labels, lookbacks, projection_length, data_shape)

        tot_per = tot_per + per
        tot_pos = tot_pos + pos
        tot_fscore = tot_fscore + fscore
        tot_accuracy = tot_accuracy + acc

        i = i + 1

    return tot_per/trials, tot_pos/trials, tot_fscore/trials, tot_accuracy/trials, predictions

"""
    testing_module()

This function is a wrapper function to all for a wide range of tests looking at varying projection lenghts and lookbacks.
This function writes the simulation settings and performance metrics in a csv file `test_results.csv`.
"""
def testing_module(trials, sentiment, feature_extraction):
    with open('test_results.csv', mode='w') as test_file:
        test_metrics_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_metrics_writer.writerow(["training_examples", "testing_examples", "offset", "lookbacks", "projection_length", "trials", "sentiment", "feature_extraction", "Performance", "% Positive", "F-Score", "Accuracy"])
        for lookbacks in range(10, 30, 5):
            for projection_length in range(10, 30, 5):
                if sentiment == 1:
                    training_examples = 100                                     # In order to get next day predictions, change this number to be the number of rows in the twitter sentiment file - the lookback length - 1 - 25
                    offset = 0
                    testing_examples = 25
                else:
                    training_examples = 2500                                    # In order to get next day predictions, change this number to be the number of rows in the financial data file - the lookback length - 1 - 1800
                    testing_examples = 900
                    offset = 900
                performance, positive, fscore, accuracy, predictions = run_trials(training_examples, testing_examples, offset, lookbacks, projection_length, trials, sentiment, feature_extraction)
                test_metrics_writer.writerow([training_examples, testing_examples, offset, lookbacks, projection_length, trials, sentiment, feature_extraction, performance, positive, fscore, accuracy])

"""
     get_next_day_predictions()

In order to actually get next day predictions the instructions on the right must be followed. These instructions cannot be done dynamically but depend on the date as well as the reloading of several datafiles.
It can be run as is and will return the last prediction made. That prediciton as the numbers are set now will be for excel row 3385 or trading day September 4th, 2018.
"""
def get_next_day_predictions():                                                 # Update Datafiles to most recent day first using Quandl, the Twitter Sentiment Generator, and the Yahoo Finance downloader
    sentiment = 0                                                               # Sentiment set = to 0, change to 1 if you want twitter sentiment.
    lookbacks = 10
    projection_length = 20
    trials = 1
    smart_feature_extraction = 0

    if sentiment == 0:
        training_examples = 3475                                                # In order to get next day predictions, change this number to be the number of rows in the financial data files - the lookback length - 1 - 900
        testing_examples = 900
        offset = 0
    else:
        training_examples = 100                                                 # In order to get next day predictions, change this number to be the number of rows in the twitter sentiment file - the lookback length - 1 - 25
        testing_examples = 25
        offset = 0

    training_sequences, test_sequences, test_ind1, test_ind2, values_2, testing_labels, data_shape = run_reward_data(training_examples, testing_examples, offset, lookbacks, projection_length, sentiment, smart_feature_extraction)
    NN_price_model = NN_price_trainer(training_sequences, lookbacks, data_shape)
    predictions = NN_predictor(NN_price_model, test_sequences)
    print(predictions[-1])

 testing_module(1, 0, 0)                        # Run without twitter sentiment and without extra feature extraction
# testing_module(1, 1, 1)                       # Run with twitter sentiment
# get_next_day_predictions()
