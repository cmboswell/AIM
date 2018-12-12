"""Equity Research
Scott Westvold
Feature Extraction File"""
################################################################################
# Module Description
################################################################################
# This module's purpose is to load in all of the market data and extract complex features from the data.
# This module calculates a number of abstractions such as moving averages and 60 day highs and reutrns a NumPy
# array of all the percent differences and percent changes extrapolated from the input data.
################################################################################
# Packages and Modules used by dataprocessing.py
################################################################################
import dataprocessing
import numpy as np

################################################################################
# Functions Used by dataprocessing.py
################################################################################

"""
    extraction(features, smart_feature_extraction)

This function takes 2 inputs. `Features` is a NumPy array which are the daily feature
descriptions for each trading day. `smart_feature_extraction` is an Int64 variable which
shoudl be a 1 or 0. When 1 this variable tells the program to extract additional features.
When 0, the relative features process will not commence to improve run-time.
This function returns a NumPy array `all_extracted_features` which contrains all the extrapolations made for each
trading day meaning that the array has the same number of rows as the `features` array but far more columns as each feature
is used to calculate and return several new abstractions including percent differce, percent difference from 60, 40, and 20 day highs and lows,
percent difference from moving average, total performance, and number of positive feature changes. 
"""
def extraction(features, smart_feature_extraction):
    change_features = np.zeros(np.shape(features))
    smart_features = np.zeros((np.shape(features)[0], 3))

    for index in range(0, np.shape(features)[0] - 1):
        change_features[index, :] = (features[index, :] - features[index - 1, :]) / (features[index - 1, :] + 1) * 100 # Multiply by 100 because the values can be very small. Make the percentages a number as opposed to a decimal.

    for index in range(1, np.shape(features)[0]):
        positive_count = 0
        index_total_percentages = 0
        sum_positive_bool = 0

        for index_2 in range(0, np.shape(features)[1] - 1):
            if change_features[index, index_2] > 0:
                positive_count = positive_count + 1                   # Number of features which were positive
                index_total_percentages = index_total_percentages + change_features[index, index_2] # Calculate total percent changes
            if index_total_percentages > 0:                           # Create a binary, was the return positive or negative variable
                sum_positive_bool = 1
        smart_features[index, :] = [positive_count, index_total_percentages, sum_positive_bool]

    change_features = np.concatenate((change_features, smart_features), 1)
    relative_features = get_relative_values(features)

    if smart_feature_extraction == 1:
        all_extracted_features = np.concatenate((change_features, relative_features), 1)
    else:
        all_extracted_features = change_features                      # A means to exclude feature extraction to improve run-time

    return all_extracted_features

"""
    `get_moving_average(features):`

This function takes 1 input of `features` which are the daily feature descriptions for each trading day.
This function then for each trading day calculates the 10 day moving average from the trailing 10 trading days.
This function returns a NumPy array `moving_average_features` which contains the daily moving averages for each trading day
for each different feature.
"""
def get_moving_average(features):
        prices = np.zeros((10, np.shape(features)[1]))
        moving_average_features = np.zeros((np.shape(features)[0], 1, 1))

        for index in range(0, np.shape(features)[0] - 1):
            for i in range(np.shape(prices)[0]-1):
                prices[i,:]= prices[i+1, :]
            prices[9, :] = features[index, :]
            moving_average_features[index, :, 0] = np.mean(prices)

        return moving_average_features

"""
    `get_Nday_high_low(features, N, high)`

This function takes 3 inputs. Features is an array of the daily feature descriptions for each trading day.
`N` is the number of trailing days over which the maximum or minimum price will be calculated for each feature on each trading day.
`high` is a boolean variable which if `True` means that the Nday highs are being calculate and Nday lows if `False`.
This function returns  an array `relative_features` which contains the daily Nday highs or lows.
"""
def get_Nday_high_low(features, N, high):
        prices = np.zeros((N, np.shape(features)[1]))
        relative_features = np.zeros((np.shape(features)[0], np.shape(features)[1], 1))

        for index in range(0, np.shape(features)[0] - 1):
            for i in range(np.shape(prices)[0] - 1):
                prices[i, :]= prices[i+1, :]
            prices[N - 1, :] = features[index, :]
            if high == True:
                relative_features[index, :, 0] = np.amax(prices, 0)
            else:
                relative_features[index, :, 0] = np.amin(prices, 0)

        return relative_features

"""
    `get_relative_prices(features, benchmark_features)`

This function finds the daily percent difference of the input data titled `features` from a numpy array of identical shape
labelled `benchmark_features`. This function returns a NumPy array of these new percent differences
titled `relative_features`.
"""
def get_relative_prices(features, benchmark_features):
    relative_features = np.zeros((np.shape(features)[0], np.shape(features)[1]))
    relative_features = features - benchmark_features[:, :, 0] / (benchmark_features[:, :, 0] + 1) * 1
    return relative_features

"""
     `get_relative_values(features)``

This function takes in an array of `features`. From these features it calls the Functions
`get_moving_average`, `get_Nday_high_low`, and `get_relative_prices` to get these descriptions
of the input features and get the daily percent differences from these extrapolated features.
This function returns a NumPy array `relative_features` of all the new features concatenated together.
"""
def get_relative_values(features):
    benchmark_1 = get_moving_average(features)
    benchmark_2 = get_Nday_high_low(features, 60, True)
    benchmark_3 = get_Nday_high_low(features, 40, True)
    benchmark_4 = get_Nday_high_low(features, 20, True)
    benchmark_5 = get_Nday_high_low(features, 60, False)
    benchmark_6 = get_Nday_high_low(features, 40, False)
    benchmark_7 = get_Nday_high_low(features, 20, False)
    relative_1 = get_relative_prices(features, benchmark_1)
    relative_2 = get_relative_prices(features, benchmark_2)
    relative_3 = get_relative_prices(features, benchmark_3)
    relative_4 = get_relative_prices(features, benchmark_4)
    relative_5 = get_relative_prices(features, benchmark_5)
    relative_6 = get_relative_prices(features, benchmark_6)
    relative_7 = get_relative_prices(features, benchmark_7)
    relative_features = np.concatenate((relative_1, relative_2, relative_3, relative_4, relative_5, relative_6, relative_7), 1)
    return relative_features
