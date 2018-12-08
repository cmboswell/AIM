#Campbell Boswell & Scott Westvold
#AIM - Automated Investment Manager
#cs701 - Senior Project
#twitterBot.py
'''
    Leverages the Tweepy user_timeline function to pull tweets posted
    on a specific user's timeline (specified in user_index.csv) during 2018.
    Stores the Tweets in a MongoDB collection that is unique for each user.
'''

import tweepy                       #primary python based twitter API
from tweepy import OAuthHandler
import config                       #holds Twitter authentication key values
import csv
from pymongo import MongoClient
import time


class TwitterBot:
    def __init__(self):
        """
        Constructor  for our TwitterBot class. Initializes connection with
        Tweepy through twitter app key/secrets stored in config.py
        """
        try:
            self.auth = OAuthHandler(config.consumer_key, config.consumer_secret)
            self.auth.set_access_token(config.access_token, config.access_secret)
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
            print("Established Connection With Twitter")
        except:
            print("Error: authentication failed")


    def get_tweets(self, offset=0):
        """
        Leverages the Tweepy user_timeline function to query for tweets posted
        on a specific user's timeline during a particular preiod.
            - The users being queried for are specified by username in the file
              referenced by config.user_list
            - We pull tweets from the users timeline for all of 2018
            - Takes an optional int argument offset, which allows the user to
              manually advance to a specific point in the user list incase the
              twitter API crashes mid run. The number of users pulled is printed
              as the program runs. This value should be used as the offset
              argument should the program crash.
        """

        #Connect to the Mongo database
        client = MongoClient('localhost', 27017) #default connection and port
        db = client['tweets'] #connect to tweets database

        #itterate through each pre-selected twitter user (whos user names are
        #stored in the file specified by config.user_list)
        with open(config.user_list) as csv_file:
            user_csv = csv.reader(csv_file, delimiter=',')

            tweets_collected = 0

            next(user_csv) #skip the first line of the file (column labels)

            #advance the csv by the number of accounts we have already pulled
            #data for - this is necissary as twitter will ocassionally throw an
            #error and require a cooldown after roughly 20,000 tweets have been
            #pulled despite being instructed to wait on a cooldown. Ideally the
            #program will not crash and just wait 15 minutes as required.
            for i in range(0, offset):
                next(user_csv)

            for row in user_csv:
                user = row[0]

                print("Collecting Tweets for @" + user)

                #connect to user's collection in our database
                collection = db[user]

                for status in tweepy.Cursor(self.api.user_timeline, id=user).items():
                    #isolate date as string in form yyyy-mm-dd
                    date = str(status.created_at).partition(" ")[0]
                    year = date[0:4]

                    #store key components of a tweet in a dictionary
                    tweet = {
                        "date": date,
                        "id": status.id,
                        "retweet_count": status.retweet_count,
                        "favorite_count": status.favorite_count,
                        "text": status.text,
                        "lang": status.lang
                    }

                    #if the tweet pre-dates 2018, stop pulling tweets
                    if year == "2017":
                        break

                    #Store the tweet object in the user collection
                    #db_tweet_id = collection.insert_one(tweet).inserted_id COMMENTED OUT WHILE TESTING THE NEWLY IMPLIMENTED AUTOMATION
                    tweets_collected += 1

                    #Print output so that we know we're making progress
                    if (tweets_collected % 100) == 0:
                        print("*", end="", flush=True)

                offset += 1
                print("")
                print("Tweets collected: " + str(tweets_collected))
                print("Number of users pulled from: " + str(offset))

            return


def main():
    # Initialize a TwitterBot object
    bot = TwitterBot()
    bot.get_tweets()

if __name__ == "__main__":
    main()
