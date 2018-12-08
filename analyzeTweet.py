#Campbell Boswell & Scott Westvold
#AIM - Automated Investment Manager
#cs701 - Senior Project
#analyzeTweet.py
'''
    Performs lexicon-based sentiment analysis using an opinion dictionary
    created by University of illinois at Chicago researchers Bing Liu and
    Minquing Hu. Seniment scores as calculated as daily positive and negative
    totals. Additionally, overall Tweet frequency is calculated and output as a
    CSV to be conveniently fed to our market analysis learning algorithm. 
'''

import config
import csv
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pymongo import MongoClient
import string
from datetime import datetime

def import_sentiment_dict():
    #append all words in our positive word dictionary to a word list
    pos_file = open('positive_words.txt', 'r')
    pos_word_list = []
    for line in pos_file:
        word = line.rstrip()
        pos_word_list.append(word)
    pos_file.close()

    #append all words in our negative word dictionary to a words list
    neg_file = open('negative_words.txt', 'r')
    neg_word_list = []
    for line in neg_file:
        word = line.rstrip()
        neg_word_list.append(word)
    neg_file.close()

    return pos_word_list, neg_word_list


def analyze_tweets():
    '''
    Performs a lexicon based analysis of tweets which have been deemed as
    relevant indicators of adjustments in the S&P 500's price. Tweets are
    scored based on the instances of either positive or negative terms in a
    dictionary complied by Minqing Hu and Bing Liu
    (http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) to determine an
    overall sentiment of the tweet. Once scored, the tweets are counted towards
    a daily sentiment score that represents the overall sentiment across the
    collection of twitter being examined.
    '''
    sentiment = {}
    #a dictionary that contains the total number of sentiment significant tweets
    #count of positive words and count of negative words for each date in our
    #twitter dataset (roughly one year of twitter data)
    pos_word_list, neg_word_list = import_sentiment_dict()

    total_tweets_analyzed = 0
    total_tweets = 0

    #open a file which will store all the words that appear in relevant tweets
    word_aggregate = open("word_aggregate.txt", "w")
    #open a file which will store all the positive words that appear
    pos_word_aggregate = open("pos_word_aggregate.txt", "w")
    #open a file which will store all the negative words that appear
    neg_word_aggregate = open("neg_word_aggregate.txt", "w")
    #open  a file which will hold user name, sentiment score and tweet text
    #which we can manually check accuracy against
    accu_file = open("manual_accuracy_test.txt", "w", encoding="utf8")

    #Connect to the Mongo database
    client = MongoClient('localhost', 27017) #default connection and port
    db = client['tweets'] #connect to raw tweets database

    #open a csv file to write sentiment scores for each user - this will be
    #used for visualizations
    per_user_csv = open('per_user_sentiment.csv', mode='w', newline='')
    per_user_sentiment = csv.writer(per_user_csv, delimiter=',')

    '''itterate through all contentes of all mongo collections'''
    with open(config.user_list) as csv_user_file:
        user_csv = csv.reader(csv_user_file, delimiter=',')

        next(user_csv) #skip the first line of the file which contains labels

        for row in user_csv:
            user = row[0]
            print("Analyzing Tweets for @" + user)

            '''basic counter vars for initial insight and visualizations'''
            user_tweets_analyzed = 0
            user_pos_tweets = 0
            user_neg_tweets = 0

            collection = db[user] #connect to user's collection in our database
            count = collection.count()
            cursor = collection.find()

            for i in range(0, count):
                doc = cursor.next()
                total_tweets += 1

                #see if this tweet was flagged as relevant (in which case it
                #will have a non-null processed_text field)
                text = doc['processed_text']
                if text == "":
                    continue

                #pull out the date of the tweet, and add it as a key in our
                #sentiment dictionary if it isn't already an element
                date = doc['date']
                if date not in sentiment:
                    sentiment[date] = [0, 0, 0]

                #itterate through all the words in the processed_text and score
                #them based on the instances of words in our sentiment dictionary
                word_tokens = word_tokenize(text)
                pos_score = 0
                neg_score = 0
                for w in word_tokens:
                    if (w in pos_word_list) or (w in neg_word_list):
                        word_aggregate.write(w + " ")
                    if (w in pos_word_list):
                        pos_score +=1
                        pos_word_aggregate.write(w + " ")
                    if (w in neg_word_list):
                        neg_score += 1
                        neg_word_aggregate.write(w + " ")

                #if there were more positive than negative words
                if pos_score > neg_score:
                    sentiment[date][1] += 1
                    user_pos_tweets += 1

                #if there were more negative than positve words
                if pos_score < neg_score:
                    sentiment[date][2] += 1
                    user_neg_tweets +=1

                #if the tweet registered as non-neutral (i.e. the pos_score and
                #neg_score were not equal), record the tweet as part of the
                #overall total of sentiment loaded tweets
                if pos_score != neg_score:
                    sentiment[date][0] += 1

                user_tweets_analyzed += 1
                total_tweets_analyzed += 1

                accu_file.write("USER :" + str(user) + "SENTIMENT SCORE: " + \
                    str(pos_score - neg_score) + " TEXT: [" + str(doc['text']) + "]\n")

            '''Printing diagnostic information and write to file'''
            print("Tweets analyzed: " + str(user_tweets_analyzed))
            print("     Positive Tweets: " + str(user_pos_tweets))
            print("     Negative Tweets: " + str(user_neg_tweets))

            #write per user sentiment data to csv to allow for the generateion
            #of visualizations
            per_user_sentiment.writerow([user, user_tweets_analyzed, \
                                         user_pos_tweets, user_neg_tweets])



    print("Total tweets analyzed: " + str(total_tweets_analyzed))
    print("Total tweets: " + str(total_tweets))

    #close all the aggregate files and per_user_csv
    word_aggregate.close()
    pos_word_aggregate.close()
    neg_word_aggregate.close()
    per_user_csv.close()
    accu_file.close()

    return sentiment

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


def store_sentiment(sentiment):
    '''Write seniment data (date, total sentiment keywords, positive sentiment
    score, negative sentiment score, positive/negative ratio)'''
    sentiment_list = []


    for key, value in sentiment.items():
        sentiment_sublist = []

        #append date to sublist
        sentiment_sublist.append(key)

        #append total sentiment keywords, pos. sentiment, neg. sentiment
        for subvalue in value:
            sentiment_sublist.append(subvalue)

        #positive tweets as a percentage
        if value[0] > 0:
            sentiment_sublist.append(value[1]/value[0])
        else:
            sentiment_sublist.append(0)


        #append sentiment sublist to sentiment list
        sentiment_list.append(sentiment_sublist)

    #write sentiment_list to csv file
    with open('sentiment.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for sublist in sentiment_list:
            writer.writerow(sublist)


def main():
    #Analyze tweets
    sentiment = analyze_tweets()
    store_sentiment(sentiment)


if __name__ == "__main__":
    main()
