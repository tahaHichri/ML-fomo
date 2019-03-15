'''
ML-Fomo ~ Written by Taha HICHRI <hishri.taha@gmail.com>, March 2019

This software is GPL licensed. The work based off of it must be released as open source.

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This file is subject to the terms and conditions defined in
file 'LICENSE.txt', which is part of this source code package.
'''
import re 
import sys
import nltk
import tweepy 
import numpy as np
from tabulate import tabulate
from textblob import TextBlob
from langdetect import detect
from tweepy import OAuthHandler 
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.parse.generate import generate, demo_grammar
from nltk.parse import ShiftReduceParser
from nltk import CFG
import language_check
from textblob.decorators import requires_nltk_corpus
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

class TwitterAnalyzer(object): 
	_classifier = None

	_checkLang = ''

	# The ignored words are twitter hashtags, links, smileys, and the search word themselves
	# This set is not final, stopwords from the wordCloud , NLTK are included also
	ignored_words = {'RT', '#', 'https', '_twt'}

	# the detected langs are sets of unique elements
	detected_langs = set()

	# words dictionary
	words = []
	search_words = []

	stop_words = []

	def __init__(self):
		self._checkLang =  language_check.LanguageTool('en-US')

		print(f'\nDownloading/fetching stopwords ..')
		nltk.download('stopwords')
		print(f'Crunching data ..\n')

		# TODO insert your Twitter API keys here
		# Create a developer account and request access
		# @link{ https://developer.twitter.com/en/apply-for-access.html} 
		consumer_key        = '<consumer_key>'
		consumer_secret     = '<consumer_secret>'
		access_token        = '<access_token>'
		access_token_secret = '<access_token_secret>'

		try: 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			self.auth.set_access_token(access_token, access_token_secret) 
			self.api = tweepy.API(self.auth)  
			# print(self.api.auth._get_request_token.value)

		except: 
			print("Error: Authentication Failed") 

		
	
	def sanitize_text(self, text):
		try:
			if detect(text) == 'en':
				allow_in_dict = True
			else:
				allow_in_dict = False
		except:
			allow_in_dict = False

		# remove non-words
		sanitized_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()) 

		self.stop_words = set(stopwords.words('english'))
		self.stop_words.update(STOPWORDS)
		self.stop_words.update(self.ignored_words)
		
		word_tokens = word_tokenize(sanitized_text) 
  
		#filtered_sentence = [w for w in word_tokens if not w in stop_words and len(w) > 1] 
  
		filtered_sentence = [] 
		# not ignored and > 1 (punctation and stuff)
		for w in word_tokens: 
		    if w not in self.stop_words and len(w) > 3 and allow_in_dict : 
		        filtered_sentence.append(w) 
		#print (filtered_sentence)

		# add words without stopwords to list
		self.words += filtered_sentence

		# I am going to need the whole text for a better classification
		return sanitized_text


	def train(self):
		super(NaiveBayesAnalyzer, self).train()
		self._classifier = nltk.classify.NaiveBayesClassifier.train(train_data)
		

	# Classify by polarity and subjectivity using TextBlob
	def get_sentiment(self, text):
		# Keep idomatic text
		text = self.sanitize_text(text)

		analysis = TextBlob(text)

		# set sentiment 
		if analysis.sentiment.polarity > 0: 
			return 'positive'
		elif analysis.sentiment.polarity == 0: 
			return 'neutral'
		else: 
			return 'negative'


	def guess_the_news(self, words):
		temp = set()
		for word in words:
			temp.add(word[0])
	
		matches = self._checkLang.check(' '.join(temp))
		
		print (f'\nHere is an auto-generated guess of what people are saying:\n')

		print (language_check.correct(' '.join(temp), matches))
		



	def fetch_tweets(self, query, count = 500): 
		# empty list to store parsed tweets 
		tweets = [] 

		# the words included in the query should be ignored from most frequently used words
		self.ignored_words.update(query.split())
		#print (self.ignored_words)
	
		try: 
			# fetch tweets 
			fetched_tweets = self.api.search(q = query, count=count) 
			# extract tweet body and guess sentiment 
			for tweet in fetched_tweets: 
				# empty dictionary for tweet, sentiment 
				parsed_tweet = {} 
				parsed_tweet['text'] = tweet.text.lower()
				parsed_tweet['sentiment'] = self.get_sentiment(tweet.text) 

				# Exclude retweets
				if tweet.retweet_count > 0: 
					if parsed_tweet not in tweets: 
						tweets.append(parsed_tweet) 
				else: 
					tweets.append(parsed_tweet) 

			# Parsed tweets 
			return tweets 

		except tweepy.TweepError as e: 
			print("Error : " + str(e)) 


def main(): 
	# creating object of TwitterClient Class 
	api = TwitterAnalyzer() 
	# calling function to get tweets 
	tweets = api.fetch_tweets(query = sys.argv[1], count = sys.argv[2] if len(sys.argv) < 2 else 500)

	# most occuring real words
	terms_occurence = Counter(api.words)
	print(f'\nMost frequently used words')
	print(terms_occurence.most_common(5))


	# picking positive tweets from tweets 
	ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive'] 
	ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']  

	positive_tweet_percentage = 100 * len(ptweets)/len(tweets)
	negative_tweet_percentage = 100 * len(ntweets)/len(tweets)
	natural_tweet_percentage  = 100 * ( len(tweets) - len(ntweets) - len(ptweets) ) / len(tweets)

	table = [["Positive",len(ptweets),positive_tweet_percentage],
	["Negative",len(ntweets),negative_tweet_percentage],
	["Neutral",( len(tweets) - len(ntweets) - len(ptweets)),natural_tweet_percentage],
	["Total", len(tweets), 100 * len(tweets)/len(tweets) ]]

	# print a grid-formatted table with stats.
	print (f'\nProcessed tweets stats (non english and REs ignored).\n')
	print(tabulate(table, headers=["Polarity","Number", "Percentage"],tablefmt="grid"))

	
	dictionary_str = ' '.join(api.words)

	api.guess_the_news(terms_occurence.most_common(15))


	# Config and show cloud of most used words
	wordcloud = WordCloud(stopwords=api.stop_words, max_font_size=40).generate(dictionary_str)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()




#one argument should be passed of type string, read on function on main
if __name__ == "__main__": 
	# calling main function 
	main() 