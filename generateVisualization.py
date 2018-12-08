#Campbell Boswell & Scott Westvold
#AIM - Automated Investment Manager
#cs701 - Senior Project
#generateVisualization.py
'''
    Generates a series of visualizations used in the final presentation and
    report for this project.
'''


import config
import csv
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pymongo import MongoClient
import string
from datetime import datetime


def generate_user_performance(num):
    '''
    Generates a visualization representing the percentage of tweets by a user
    which were deemed relevant by our filtering process. Also produces a
    visualization which compares the number of positive versus negative tweets
    that each user post.
    '''

    relevance = []
    sentiment = []
    users = []

    #we will reduce the full lists down to top and bottom for each catagory
    most_relevant_value = []
    most_relevant_user = []
    least_relevant_value = []
    least_relevant_user = []
    most_sentiment_value = []
    most_sentiment_user = []
    least_sentiment_value = []
    least_sentiment_user = []

    with open('per_user_relevance.csv', mode='r') as relevance_csv,\
         open('per_user_sentiment.csv', mode='r') as sentiment_csv:

        r_reader = csv.reader(relevance_csv, delimiter=',')
        s_reader = csv.reader(sentiment_csv, delimiter=',')

        #build lists from csv data
        for r_row in r_reader:
            users.append(r_row[0])
            relevance.append(float('%.4f'%(float(r_row[1]))))

        for s_row in s_reader:
            sentiment.append((int(s_row[2])/(int(s_row[2])+int(s_row[3])))*100)

        if num > (len(relevance) / 2):
            print("Requested num of output values greater than total num values")
            return

        #ooooof, this is a very conveluded way to achieve this, but there doesn't
        #seem to be a better way because we need to mainitain the integrity of
        #index values across the user and value lists
        temp_relevance = relevance.copy()
        temp_sentiment = sentiment.copy()
        temp_users = users.copy()

        #generate max and min lists for relevance
        for i in range(num):
            #generate max value lists
            index = relevance.index(max(relevance))
            value = relevance[index]
            user = users[index]
            del relevance[index]
            del users[index]
            most_relevant_value.append(value)
            most_relevant_user.append(user)

            #generate min value lists
            index = relevance.index(min(relevance))
            value = relevance[index]
            user = users[index]
            del relevance[index]
            del users[index]
            least_relevant_value.append(value)
            least_relevant_user.append(user)

        relevance = temp_relevance
        sentiment = temp_sentiment
        users = temp_users

        #generate max and min lists for sentiment
        for i in range(num):
            #generate max value lists
            index = sentiment.index(max(sentiment))
            value = sentiment[index]
            user = users[index]
            del sentiment[index]
            del users[index]
            most_sentiment_value.append(value)
            most_sentiment_user.append(user)

            #generate min value lists
            index = sentiment.index(min(sentiment))
            value = sentiment[index]
            user = users[index]
            del sentiment[index]
            del users[index]
            least_sentiment_value.append(value)
            least_sentiment_user.append(user)

    #build the percent relevant visualizations
    fig, ax = plt.subplots(figsize=(12,8))
    most = plt.bar(most_relevant_user, most_relevant_value, edgecolor='teal',\
            facecolor='cadetblue')
    least = plt.bar(least_relevant_user, least_relevant_value, edgecolor='teal',\
            facecolor='cadetblue')

    #prints values of the bars in our charts above the bars
    autolabel(most, ax, most_relevant_value)
    autolabel(least, ax, least_relevant_value)
    plt.xlabel('Users')
    plt.ylabel('Percent Tweets Relevant', color='black')
    plt.title('Relevance of User Tweets')
    plt.tight_layout()
    #save the figure we generated as a .png file
    plt.savefig('relevance.png', bbox_inches='tight')

    #build the sentment visualizations
    fig, ax = plt.subplots(figsize=(12,8))
    most = plt.bar(most_sentiment_user, most_sentiment_value, edgecolor='teal',\
            facecolor='cadetblue')
    least = plt.bar(least_sentiment_user, least_sentiment_value, edgecolor='teal',\
            facecolor='cadetblue')

    #prints values of the bars in our charts above the bars
    autolabel(most, ax, most_sentiment_value)
    autolabel(least, ax, least_sentiment_value)
    plt.xlabel('Users')
    plt.ylabel('Percent Positve Tweets', color='black')
    plt.title('Sentiment of User Tweets')
    plt.tight_layout()

    #save the figure we generated as a .png file
    plt.savefig('seniment.png', bbox_inches='tight')




def generate_price_vs_sentiment():
    '''A function that creates plots of the sp500's price vs the sentiment for a
    given date. We can create multiple visualizations in this function which
    seem to be the most informative'''

    sentiment_date = []
    num_pos_tweets = []
    num_neg_tweets = []

    #daily prices of the sp500 based on the index open price
    sp_price = []

    #store values from sentiment.csv in lists organized in order of accending
    #date to allow for us to filter out non-trading days when importing sp500 data
    #NOTE THAT IS IS NECESSARY TO OPEN sentment.csv IN EXCEL AND SORT MANUALLY
    #BY "OLDEST TO NEWEST"
    with open("sentiment.csv") as csv_file:
        sentiment_csv = csv.reader(csv_file, delimiter=',')

        for row in sentiment_csv:
            date = format_sentiment_datetime(row[0])
            sentiment_date.append(date)
            num_pos_tweets.append(int(row[2]))
            num_neg_tweets.append(int(row[3])) #negate negative tweet count

    with open("sp500_price_2018.csv") as csv_file:
        sp_csv = csv.reader(csv_file, delimiter=',')

        row = next(sp_csv) #Initializes a pointer to the first row in the csv
        row = next(sp_csv) #skips the first line of the csv which contains lable

        #Itterate throught the sentinent_date list and read through the
        #sp500 price date starting with January 1, 2018. Store the opening price
        #in the list sp_price and remove the sentinent data from the list if the
        #date in sentinent_data isn't a valid trading day
        i = 0
        while i < len(sentiment_date):

            #if it's a non-trading day ignore the sentment for the purpose of
            #the visualizations that we're building...
            date_str = str(row[0]).replace("-", "")
            sp_date = datetime.strptime(date_str, "%Y%m%d")

            if sentiment_date[i] != sp_date:
                #print("dates don't lineup..." + "sp500 date: "+ str(sp_date) + " sentment date: " + str(sentiment_date[i]))
                del sentiment_date[i]
                del num_pos_tweets[i]
                del num_neg_tweets[i]

            else:
                open_price = float(row[1])
                close_price = float(row[4])
                sp_price.append(abs(open_price))

                #try to advance to the next element in the sp500 csv, if it
                #throws an exception, we should return
                try:
                    row = next(sp_csv)
                except:
                    #trim the trailing values in sentinent data lists
                    sentiment_date = sentiment_date[0:len(sp_price)]
                    num_pos_tweets = num_pos_tweets[0:len(sp_price)]
                    num_neg_tweets = num_neg_tweets[0:len(sp_price)]
                    break
                i+=1

    #create the visualization from the list of prices and sentiment scores
    print("Number of sentiment values: (pos) "+str(len(num_pos_tweets)) + " (neg) " +str(len(num_neg_tweets)))
    print("Number of sp500 values: "+str(len(sp_price)))

    num_total_tweets = []
    for i in range(len(num_pos_tweets)):
        num_total_tweets.append(num_neg_tweets[i] + num_pos_tweets[i])

    #generate a combined sentiment score that is the daily overall sentiment
    #normalized by the daily sentiment total to give a percentage
    combined_sentiment = []
    for i in range(len(sentiment_date)):
        combined_sentiment.append((num_pos_tweets[i] - num_neg_tweets[i])/ \
                                  (num_pos_tweets[i] + num_neg_tweets[i]))


    #plot s&p data on axis 1
    fig, ax1 = plt.subplots(figsize=(12,8))
    ax1.plot(sentiment_date, sp_price, color='black', marker='.')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Opening Price of S&P 500 ($)', color='black')

    #plot sentinent data on axis 2
    ax2 = ax1.twinx()
    ax2.bar(sentiment_date, combined_sentiment, edgecolor='cadetblue',\
        facecolor='mediumturquoise')
    ax2.set_ylabel('Sentiment', color='cadetblue')
    plt.title('S&P 500 Price vs. Twitter Sentiment')
    fig.tight_layout()

    #save the figure we generated as a .png file
    plt.savefig('sp_vs_seniment.png', bbox_inches='tight')


def generate_wordcloud():
    '''A function which uses the wordcloud library to create visualizations
    that show the frequency of terms in the aggregate word files that were
    compiled in the analyze_tweets function'''

    #open word aggregate files
    word_aggregate_file = open("word_aggregate.txt", "r")
    word_aggregate = word_aggregate_file.read()

    pos_word_aggregate_file = open("pos_word_aggregate.txt", "r")
    pos_word_aggregate = pos_word_aggregate_file.read()

    neg_word_aggregate_file = open("neg_word_aggregate.txt", "r")
    neg_word_aggregate = neg_word_aggregate_file.read()

    #create wordclouds
    aggregate_cloud = WordCloud(width=1600, height=800, colormap="Blues", background_color="white").generate(word_aggregate)
    pos_cloud = WordCloud(width=1600, height=800, colormap="Greens", background_color="white").generate(pos_word_aggregate)
    neg_cloud = WordCloud(width=1600, height=800, colormap="Reds", background_color="white").generate(neg_word_aggregate)

    #store wordclouds as image files
    aggregate_cloud.to_file("aggregate_word_cloud.png")
    pos_cloud.to_file("pos_word_cloud.png")
    neg_cloud.to_file("neg_word_cloud.png")

    #close aggregate files
    word_aggregate_file.close()
    pos_word_aggregate_file.close()
    neg_word_aggregate_file.close()


def format_sentiment_datetime(input_date):
    '''A basic function to properly format the poorly formated date values that
    were taken from the twitter sentiment data'''
    date_str = str(input_date)
    date_list = []
    num_digits = 0

    for i in range(len(date_str)):
        if date_str[i] == "/":
            if num_digits == 1:
                date_list.insert(0, "0")
                num_digits += 1
            if num_digits == 3:
                date_list.insert(2, "0")
                num_digits += 1
        else:
            date_list.append(date_str[i])
            num_digits += 1

    return datetime.strptime(''.join(date_list),"%m%d%Y")

def autolabel(rects, ax, label_values):
    '''
    Function which allows us to print the values of bars above the bars in our
    bar charts. This code is a variation on matplotlib sample code in matplotlib
    API and can should be attributed to Lindsey Kuper and her blog composition.al
    '''
    #Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    i = 0
    for rect in rects:
        height = rect.get_height()
        label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                '%.2f' % float(label_values[i]),
                ha='center', va='bottom')
        i += 1


def main():
    generate_wordcloud()
    #generate_price_vs_sentiment()
    #generate_user_performance(5)

if __name__ == "__main__":
    main()
