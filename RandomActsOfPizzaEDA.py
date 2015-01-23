#******************************************************
# Author: Bill Zichos
#******************************************************


#******************************************************
# Objective: Explore the data for the Random Acts of
#     Pizza competition.
#******************************************************


#******************************************************
# Module import
#******************************************************
import json
import nltk
from nltk.probability import FreqDist
#import csv
#import os
import re


#******************************************************
# Functions
#******************************************************
def bzSentenceCount(text):
        return len(nltk.sent_tokenize(text))

def bzWordCount(text):
        return len(nltk.word_tokenize(text))

def bzLexicalDiversity(text):
        return len(set([words.lower() for words in nltk.word_tokenize(text)])) / len(nltk.word_tokenize(text))


#******************************************************
# Read in the training and test files
#******************************************************
filepath = 'C:\\Users\\Bill\\Documents\\GitHub\\Project-Files\\Kaggle - Random Acts of Pizza Train.json'
testfilepath = 'C:\\Users\\Bill\\Documents\\GitHub\\Project-Files\\Kaggle - Random Acts of Pizza Test.json'

json_data = open(filepath)
json_test_data = open(testfilepath)

data = json.load(json_data)
test_data = json.load(json_test_data)


#******************************************************
# Add features to the datasets
#******************************************************
for item in data:
	if len(item['request_text'])>0:
		item['SentCount']=bzSentenceCount(item['request_text'])
		item['WordCount']=bzWordCount(item['request_text'])
		item['LexicalDiversity']=bzLexicalDiversity(item['request_text'])

picList = [item['request_id'] for item in data if re.search('.jpg|.png', item['request_text'])]

for item in data:
	if item['request_id'] in picList:
		item['Picture']=1
	else:
		item['Picture']=0


#len([item for item in data if item['RequestTextFlag']==0])


## Let's explore differences b/w successful requests and unsuccessful.
print('Begin calculating conversion rate...')

goodRequests = [item for item in data if item['requester_received_pizza']==1]
badRequests = [item for item in data if item['requester_received_pizza']==0]

print('...' + str((len([item for item in data if item['requester_received_pizza']==1]) / len([item for item in data])) * 100) + '% of requests are fullfilled.') 



#*******************************************************************************
# Question: Is there anything interesting about word counts?
# Answer: Yes
#*******************************************************************************
print('Begin calculating average word counts...')

goodAvgWordCnt = sum([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1]) / len([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1])
print('...Average word count for converted requests: ' + str(goodAvgWordCnt))
badAvgWordCnt = sum([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0]) / len([item['WordCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0])
print('...Average word count for unconverted requests: ' + str(badAvgWordCnt))



#*******************************************************************************
# Question: Is there anything interesting about sentence counts?
# Answer: Yes
#*******************************************************************************
print('Begin calculating average sentence counts...')

goodAvgSentCnt = sum([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1]) / len([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1])
print('...Average sentence count for converted requests: ' + str(goodAvgSentCnt))
badAvgSentCnt = sum([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0]) / len([item['SentCount'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0])
print('...Average sentence count for unconverted requests: ' + str(badAvgSentCnt))



#*******************************************************************************
# Question: Is there anything interesting about lexical diversity?
# Answer: Marginal
#*******************************************************************************
print('Begin calculating lexical diversity...')

goodAvgLexDty = sum([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1]) / len([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==1])
print('...Average lexical diversity for converted requests: ' + str(goodAvgLexDty))
badAvgLexDty = sum([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0]) / len([item['LexicalDiversity'] for item in data if len(item['request_text'])>0 and item['requester_received_pizza']==0])
print('...Average lexical diversity for unconverted requests: ' + str(badAvgLexDty))



#*******************************************************************************
# Question: Are there differences between word-length frequencies of converted
#      vs. unconverted requests?
# Answer: No
#*******************************************************************************
print('Begin calculating word length frequencies...')
      
cnvtText = ' '.join([item['request_text'] for item in data
                     if len(item['request_text'])>0
                     and item['requester_received_pizza']==1])
wl1 = [len(word) for word in nltk.word_tokenize(cnvtText) if word.isalpha()]
wl1fd = FreqDist(wl1)
wl1fd.plot()
## 4, 3, 2, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 18
print('...Word length frequencies for successful requests have been plotted.')

uncnvtText = ' '.join([item['request_text'] for item in data
                     if len(item['request_text'])>0
                     and item['requester_received_pizza']==0])
wl2 = [len(word) for word in nltk.word_tokenize(uncnvtText) if word.isalpha()]
wl2fd = FreqDist(wl2)
wl2fd.plot()
## 4, 3, 2, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 17, 35, 20
print('...Word length frequencies for unsuccessful requests have been plotted.')



#*******************************************************************************
# Question: Do requests without text ever get converted?
# Answer: Yes, >17%
#*******************************************************************************
print('Do requests without text ever get converted?')

noTextCount = len([item['requester_received_pizza'] for item in data if len(item['request_text'])==0])
noTextSum = sum([item['requester_received_pizza'] for item in data if len(item['request_text'])==0])

print(noTextSum / noTextCount)




#*******************************************************************************
# Question: Are curse words a turnoff?
# Answer:
#*******************************************************************************
print('Are curse words a turnoff?')







## Let's evaluate frequently occuring, large words across the two datasets.

print('Begin important word search...')

FreqDist([word.lower() for word in nltk.word_tokenize(cnvtText) if word.isalpha() and len(word) > 6]).plot(50)
FreqDist([word.lower() for word in nltk.word_tokenize(uncnvtText) if word.isalpha() and len(word) > 6]).plot(50)

## I see some patterns around cravings, school and employment.
## I should probably explore keywords around these themes.
## CRAVINGS - crave, craving, love, party, hangover, friend, friends, girlfriend, boyfriend, night, coming, late, starving, starve, birthday
## SCHOOL - school, class, college, university, study, exam, exams, test, cram, student, practice, team, reading, read
## EMPLOYMENT - job, work, unemployed, lost, unemployment, paycheck, check, cash, account, provide, husband, wife

## I also see themes around gratitude and politeness
## appreciate, thanks, greatly, grateful,


## Pictures, attachments, etc?

[item['request_id'] for item in data if re.search('.jpg|.png', item['request_text'])]


## Let's see what we can learn about the individual requesters.

## How many unique requestors are there?
print('Begin unique requester analysis...')
## 4040 out of 5671
print('...There are ' + str(len(set([item['requester_username'] for item in data]))) + ' unique users.')










## Export a CSV file for training an algorithm.

import csv
import sys

with open("C:\\Users\\Bill\\SkyDrive\\Documents\\Kaggle\\Random Acts of Pizza\\presubmission.csv", "w", newline="") as presub:
	presubwrite = csv.writer(presub, delimiter=',')
	presubwrite.writerows([item['requester_username'] for item in goodRequests])


##pizzaRequests = {'RequestId': '', 'PizzaYN': False, 'Request': '', 'SentenceCount': 0, 'WordCount': 0, 'LexicalDiversity': 0}
##
##for item in data:
##    pizzaRequests['RequestId'] = item['request_id']
##    pizzaRequests['PizzaYN'] = item['requester_received_pizza']
##    pizzaRequests['Request'] = item['request_text']
##    if len(item['request_text']) > 0:
##        pizzaRequests['SentenceCount'] = len(nltk.sent_tokenize(item['request_text']))
##        pizzaRequests['WordCount'] = len(nltk.word_tokenize(item['request_text']))
##        pizzaRequests['LexicalDiversity'] = len(set([words.lower() for words in nltk.word_tokenize(item['request_text'])])) / pizzaRequests['WordCount']

##{item['SentCount']: len(nltk.sent_tokenize(item['request_text'])) for item in data}
##{item['WordCount']: len(nltk.word_tokenize(item['request_text'])) for item in data if len(item['request_text'])>0}
##{item['LexicalDiversity']: len(set([words.lower() for words in nltk.word_tokenize(item['request_text'])])) / len(nltk.word_tokenize(item['request_text'])) for item in data if len(item['request_text'])>0}





##text2 = {'requester_number_of_comments_at_request': 0, 'request_text_edit_aware': 'Hi I am in need of food for my 4 children we are a military family that has really hit hard times and we have exahusted all means of help just to be able to feed my family and make it through another night is all i ask i know our blessing is coming so whatever u can find in your heart to give is greatly appreciated', 'number_of_upvotes_of_request_at_retrieval': 1, 'giver_username_if_known': 'N/A', 'request_id': 't3_l25d7', 'requester_username': 'nickylvst', 'requester_upvotes_minus_downvotes_at_retrieval': 1, 'requester_received_pizza': False, 'requester_number_of_subreddits_at_request': 0, 'requester_subreddits_at_request': [], 'request_text': 'Hi I am in need of food for my 4 children we are a military family that has really hit hard times and we have exahusted all means of help just to be able to feed my family and make it through another night is all i ask i know our blessing is coming so whatever u can find in your heart to give is greatly appreciated', 'requester_number_of_posts_at_retrieval': 1, 'requester_days_since_first_post_on_raop_at_request': 0.0, 'requester_number_of_posts_at_request': 0, 'requester_number_of_comments_in_raop_at_request': 0, 'number_of_downvotes_of_request_at_retrieval': 0, 'requester_number_of_comments_at_retrieval': 0, 'unix_timestamp_of_request_utc': 1317849007.0, 'requester_account_age_in_days_at_retrieval': 792.4204050925925, 'requester_user_flair': None, 'requester_account_age_in_days_at_request': 0.0, 'requester_number_of_comments_in_raop_at_retrieval': 0, 'requester_number_of_posts_on_raop_at_retrieval': 1, 'unix_timestamp_of_request': 1317852607.0, 'request_number_of_comments_at_retrieval': 0, 'request_title': 'Request Colorado Springs Help Us Please', 'requester_number_of_posts_on_raop_at_request': 0, 'requester_days_since_first_post_on_raop_at_retrieval': 792.4204050925925, 'requester_upvotes_minus_downvotes_at_request': 0, 'post_was_edited': False, 'requester_upvotes_plus_downvotes_at_retrieval': 1, 'requester_upvotes_plus_downvotes_at_request': 0}
##
##sents = nltk.sent_tokenize(text)
##sentCount = len(sents)
##
##words = nltk.word_tokenize(text)
##wordCount = len(words)
##
##lexicalDiversity = len(set(words.lower() for words in text)) / wordCount
