A.I.M. - Automated Investment Manager
A senior project for the Middlebury College Computer Science Department
completed by Campbell Boswell and Scott Westvold

1. Description
With recent advancements in deep learning, neural networks have become extremely adept at pattern fitting and recognition. One discipline where pattern detection is central to advantageous outcomes and the derivation of insight is financial  market research and analysis. Over the course of the Fall 2018 semester we examined the potential for deep learning to predict market events by combining  neural networks with sentiment values derived from Twitter posts. This repository contains the source code from the project, as well as files necessary to reproduce our Twitter dataset.

2. Prerequisites
DO WE NEED A FULL LIST OF PACKAGES AND DETAILS ON HOW TO BUILD AND RUN OUR SOFTWARE???

3. Contents

twitterBot.py -     
    Leverages the Tweepy user_timeline function to pull tweets posted on a specific user's timeline (specified in user_index.csv) during 2018.
    Stores the Tweets in a MongoDB collection that is unique for each user.

processTweet.py -
    Filters Tweets based on content so that only relevant tweets are analyzed and scored for sentiment. The filtered Tweets are defined as those which contain the name or ticker symbol of a company listed on the S&P 500 or Tweets which contain a reference to the market in general (as defined in market_keywords.py).

analyzeTweet.py -
    Performs lexicon-based sentiment analysis using an opinion dictionary created by University of Illinois at Chicago researchers Bing Liu and Minquing Hu. Sentiment scores as calculated as daily positive and negative totals. Additionally, overall Tweet frequency is calculated and output as a CSV to be conveniently fed to our market analysis learning algorithm.

config.py -
    A file containing the access keys for the Twitter bot we use to access the Twitter API.

generateVisualization.py -
    Generates a series of visualizations used in the final presentation and report for this project.

company_filter.py -
    A list with a series of common words that are included in the names of corporations listed on the S&P 500. These words do not count as keywords which can distinguish a Tweet as relevant

constituents_filter_except.py -
    A dictionary with a series of first words from the names of companies on the S&P 500. The value pairs are the subsequent words in the company names. This dictionary is used during the filtering process.   

market_keywords.py -
    A series of keywords used in the filtering process to indicate that a Tweet
    is relevant to overall market trends or sentiment.

negative_words.txt
    The negative word sentiment dictionary generated by Bing Liu and Minquing Hu.

positve_words.txt
    The positive word sentiment dictionary generated by Bing Liu and Minquing Hu.

user_index.csv
    A list of the Twitter usernames for the users who comprise our sample set.