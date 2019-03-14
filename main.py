# I have deliberatly chosen this order :p
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
from textblob.decorators import requires_nltk_corpus

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

class TwitterAnalyzer(object): 
	_classifier = None

	
	# The ignored words are twitter hashtags, links, smileys, and the search word themselves
	ignored_words = {'RT', '#'}

	# the detected langs are sets of unique elements
	detected_langs = set()

	# words dictionary
	words = []
	search_words = []

	stop_words = []

	def __init__(self):
		
		print(f'\nDownloading/fetching stopwords ..')
		nltk.download('stopwords')
		print(f'Crunching data ..\n')
		consumer_key        = '28DygxxutAjv0sgfdETPaWrDu'
		consumer_secret     = '2d5Fn5ez4rXDue1S5kEXLrBqwsvTsCszcahLPUkfEV1UiBnodD'
		access_token        = '715467695093886976-JtC3EeyRYDie0TUvsndL0uya4aavjLI'
		access_token_secret = 'nxew0AHqnCET1wyXDYYgN9w0SLR4x8yPsqW070kyfsLfj'

		try: 
			self.auth = OAuthHandler(consumer_key, consumer_secret) 
			self.auth.set_access_token(access_token, access_token_secret) 
			self.api = tweepy.API(self.auth)  
			# print(self.api.auth._get_request_token.value)

		except: 
			print("Error: Authentication Failed") 

		
	

	def sanitize_text(self, text):
		
		if detect(text) == 'en':
			allow_in_dict = True
		else:
			allow_in_dict = False

		# remove non-words
		sanitized_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split()) 


		# self.detected_langs.add(TextBlob(sanitized_text).detect_language())

		# TODO stop words language can be set based on the detected language of the tweet.
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
		temp = []
		for word in words:
			temp +=word[0]
		 
		blob_from_most_used = TextBlob(' '.join(temp))
		blob_from_most_used.ngrams(n=5)
		print (blob_from_most_used)




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
	tweets = api.fetch_tweets(query = sys.argv[1], count = sys.argv[2])

	# most occuring real words
	terms_occurence = Counter(api.words)
	print(f'\nMost frequently used words')
	print(terms_occurence.most_common(10))

	#print(terms_occurence)

	# print(f'\nMost langs')
	# print(api.detected_langs)

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

	api.guess_the_news(terms_occurence.most_common(10))

	# Generate a word cloud image
	# wordcloud = WordCloud(stopwords=api.ignored_words).generate(dictionary_str)

	# plt.imshow(wordcloud, interpolation='bilinear')
	# plt.axis("off")

	# lower max_font_size
	wordcloud = WordCloud(stopwords=api.stop_words, max_font_size=40).generate(dictionary_str)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()


	

	# wc = WordCloud(background_color="white", max_words=2000, mask=alice_mask,
    #            stopwords=stopwords, contour_width=3, contour_color='steelblue')




#one argument should be passed of type string, read on function on main
if __name__ == "__main__": 
	# calling main function 
	main() 
