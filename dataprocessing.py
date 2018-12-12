"""Equity Research
Scott Westvold
Data Processing File"""
################################################################################
# Module Description
################################################################################
# This module's purpose is to load in all of the market data and convert it into
# Numpy arrays which can be processed for model training and testing. This Module
# zeros out data in the incorrect format. This module assigns 1, 0 values to
# input data as labels representing the S and P going up or down respectively.
################################################################################
# Packages and Modules used by dataprocessing.py
################################################################################
import numpy as np
import csv
import feature_extraction as fe

################################################################################
# Functions Used by dataprocessing.py
################################################################################
"""
    `load_data(examples, filename, sentiment)

This function takes in 1 csv file, represented by `filename` and processes through
`examples` number of rows of that csv file in order to get input data into a numpy
array to return. The returned numpy array `features` will be of dimensions `examples` by features
(columns of the numpy array) by 1 (for data formatting).
"""
def load_data(examples, filename, sentiment):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting = csv.QUOTE_MINIMAL)
        index = 0                                                     # Index in my returned numpy array
        first_row = 0                                                 # Indicator variable to avoid the column lables in the csvfile
        if sentiment == 0:
            starting_date = 1
        else:
            starting_date = 4263                                      # for sentiment_data starting at a later date
        for row in reader:
            if index < examples:
                if first_row == 0:
                    features = np.zeros((examples, np.shape(row)[0] - 1, 1))    # Declare Numpy array to store data
                    first_row = 1
                elif first_row < starting_date:
                    first_row = first_row + 1
                else:
                    for value in range(np.shape(row)[0]):
                        if row[value] == '':                          # Incomplete data, 0 out
                            row[value] = 0
                    features[index, :, 0] = row[1:]
                    index = index + 1
        return features

"""
    `load_sentiment_data(examples, filename)

This function takes in 1 csv file, represented by `filename` and processes through
`examples` number of rows of that csv file in order to get input data into a numpy
array to return. The returned numpy array `features` will be of dimensions `examples` by features
(columns of the numpy array) by 1 (for data formatting). This function is specifically purposed to deal with
twitter sentiment data which is weekly rather than operating on trading days only.
"""
def load_sentiment_data(examples, filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting = csv.QUOTE_MINIMAL)
        index = 0                                                     # Index in my returned numpy array
        first_row = 0                                                 # Indicator variable to avoid the column lables in the csvfile
        day_of_week = 0                                               # 0 -> 6 represents Monday through Sunday
        for row in reader:
            if index < examples:
                if first_row == 0:
                    features = np.zeros((examples, np.shape(row)[0] - 1, 1))    # Declare Numpy array to store data
                    first_row = 1
                else:
                    for value in range(np.shape(row)[0]):
                        if row[value] == '':                          # Incomplete data, 0 out
                            row[value] = 0
                    if day_of_week < 5:
                        features[index, :, 0] = row[1:]
                        index = index + 1
                        day_of_week += 1
                    elif day_of_week == 5:
                        day_of_week += 1
                    else:
                        day_of_week = 0
        return features

"""
    `all_input_data(filenames, examples, sentiment)

This function takes in an array of `filenames` which it will loop over and will
call on `load_data()` to retrieve `examples` rows of each csv file in the array.
This array will by built out to be `examples` by the sum of all the features in
each csv file. This array `input_array_final` containing all the input features will
be returned.
"""
def all_input_data(filenames, examples, sentiment):
    hardcoded_total_features = 1500                                   # For memory allocation
    feature_index = 0                                                 # So that the memory used can be kept track of
    input_array = np.zeros((examples, hardcoded_total_features, 1))

    for file in filenames:
        if file == "sentiment2.csv":
            single_input_array = load_sentiment_data(examples, file)
        else:
            single_input_array = load_data(examples, file, sentiment)
        number_features = np.shape(single_input_array)[1]
        input_array[:, feature_index:feature_index + number_features, :] = single_input_array
        feature_index = feature_index + number_features

    input_array_final = input_array[:, :feature_index, 0]             # Only Return Used Memory
    return input_array_final

"""
    `get_input_labels(filename, examples, lookbacks, projection_length, sentiment)

This function creates the labels for the training and testing data. It will input a
1 into the `labels` array if the index increases over `projection_length` and 0 if
it does not. It also returns an array `stock_returns` which contains the return on investment as a decimal if the S & P
were bought that day and sold `projection_length` days later.
"""
def get_input_labels(filename, examples, lookbacks, projection_length, sentiment):
    labels = np.zeros((examples, 1))
    stock_returns = np.zeros((examples, 1))
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, quoting = csv.QUOTE_MINIMAL)
        index = 0
        first_row = 0                                                 # Indicator for avoiding the column headers
        if sentiment == 0:
            starting_date = 1
        else:
            starting_date = 4263                                      # for sentiment_data starting at a later date
        for row in reader:
            if index < examples:
                if first_row < lookbacks + 1:
                    first_row = first_row + 1
                    prices = np.zeros((projection_length, 1))
                elif first_row < starting_date:
                    first_row = first_row + 1
                else:
                    for value in range(np.shape(row)[0]):
                        if row[value] == '':                          # test for incomplete data and 0 it out
                            row[value] = 0

                    for i in range(np.shape(prices)[0] - 1):          # Shift array of prices so that the `proejction length` most recent prices are in the array in order
                        prices[i, :]= prices[i+1, :]

                    prices[projection_length - 1, :] = row[4]
                    first_row = first_row + 1

                    if first_row > lookbacks + projection_length:     # I need to wait through the lookback data as it is input data and will not be used as labels
                        stock_returns[index, 0] = (prices[projection_length - 1, 0] - prices[0, 0])/prices[0, 0]
                        if prices[0, 0] < prices[projection_length - 1, 0]:
                            labels[index, 0] = 1                      # S & P increase gets a label of 1
                        else:
                            labels[index, 0] = 0
                        index = index + 1
        return labels, stock_returns


"""
    `run_program(examples, lookbacks, projection_length, sentiment, smart_feature_extraction)`

This function initializes and calls for all data collection to get all data features
and labels. It returns 2 arrays, `input_data` and `input labels` whose rows correspond
with one another for each training/testing example. It also returns an array `stock_returns`
"""
def run_program(examples, lookbacks, projection_length, sentiment, smart_feature_extraction):
    if sentiment == 0:
        input_data = all_input_data(["gspc.csv", "financials.csv", "totals.csv", "30yearTbill.csv", "10yearTbill.csv", "2yearTbill.csv"], examples, sentiment)
    else:                                                             # Only process sentiment data if user input says to
        input_data = all_input_data(["gspc.csv", "financials.csv", "totals.csv", "sentiment2.csv", "30yearTbill.csv", "10yearTbill.csv", "2yearTbill.csv"], examples, sentiment)
    second_input = fe.extraction(input_data, smart_feature_extraction)
    input_labels, stock_returns = get_input_labels("gspc.csv", examples, lookbacks, projection_length, sentiment)
    input_data = second_input
    return input_data, input_labels, stock_returns
