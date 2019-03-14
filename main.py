import re 
import sys
import tweepy 
import numpy as np
from textblob import TextBlob 
from tweepy import OAuthHandler 
from collections import Counter

class TwitterAnalyzer(object): 
	words = []
	search_words = []
	''' 
	Generic Twitter Class for sentiment analysis. 
	'''
	def __init__(self):
		consumer_key        = 'C9Zya8TrjZrd9kxFAmcSPT50k'
		consumer_secret     = 'jhIkCmWOJi1PG3XNRpQIHHpb8AlArmAwNh0BlCH00lXJxEVL1P'
		access_token        = '715467695093886976-a0cwx3FnD4HV1rwuJMp2YxwoWMuqKCG'
		access_token_secret = 'ZfPvgFFBpjKbqrO1OIpZtuGEyisTSXlZOsXkIpLnqxndM'

		# attempt authentication 
		try: 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			self.auth.set_access_token(access_token, access_token_secret) 
			self.api = tweepy.API(self.auth) 
		except: 
			print("Error: Authentication Failed") 


	''' 
	Utility function to clean tweet text by removing links, special characters 
	using simple regex statements. 
	'''
	def clean_tweet(self, tweet):
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 

	def get_tweet_sentiment(self, tweet): 
		# create TextBlob object of passed tweet text 
		clean_tweet = self.clean_tweet(tweet)
		analysis = TextBlob(clean_tweet) 

		# add words to words list dictionnary
		self.words += clean_tweet.split()

		# set sentiment 
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'

	def get_tweets(self, query, count = 100): 
		''' 
		Main function to fetch tweets and parse them. 
		'''
		# empty list to store parsed tweets 
		tweets = [] 

		try: 
			# call twitter api to fetch tweets 
			fetched_tweets = self.api.search(q = query, count = count) 
			# parsing tweets one by one 
			for tweet in fetched_tweets: 
				# empty dictionary to store required params of a tweet 
				parsed_tweet = {} 

				# saving text of tweet 
				parsed_tweet['text'] = tweet.text 
				# saving sentiment of tweet 
				parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text) 

				# appending parsed tweet to tweets list 
				if tweet.retweet_count > 0: 
					# if tweet has retweets, ensure that it is appended only once 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			# return parsed tweets 
			return tweets 

		except tweepy.TweepError as e: 
			# print error (if any) 
			print("Error : " + str(e)) 

def main(): 
	# creating object of TwitterClient Class 
	api = TwitterAnalyzer() 
	# calling function to get tweets 
	tweets = api.get_tweets(query = sys.argv[1], count = 200)

	terms_occurence = Counter(api.words)
	print(terms_occurence.most_common(5))

	# picking positive tweets from tweets 
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	# percentage of positive tweets 
	print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))) 
	# picking negative tweets from tweets 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative'] 
	# percentage of negative tweets 
	print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))) 
	# percentage of neutral tweets 
	print("Neutral tweets percentage: {} % ".format(100* (len(tweets) - len(ntweets) - len(ptweets))/len(tweets))) 

	# printing first 5 positive tweets 
	# print("\n\nPositive tweets:") 
	# for tweet in ptweets[:10]: 
	# 	print(tweet['text']) 

	# # printing first 5 negative tweets 
	# print("\n\nNegative tweets:") 
	# for tweet in ntweets[:10]: 
	# 	print(tweet['text']) 


#one argument should be passed of type string, read on function on main
if __name__ == "__main__": 
	# calling main function 
	main() 


    # wiki = TextBlob("Python is a high-level, language general-purpose programming language.")
    # print (type(wiki.words))