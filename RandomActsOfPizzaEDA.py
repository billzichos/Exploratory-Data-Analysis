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
import re
import numpy


#******************************************************
# Functions
#******************************************************
graphs = 'no' # print the graphs?

#******************************************************
# Functions
#******************************************************
def bzSentenceCount(text):
        return len(nltk.sent_tokenize(text))

def bzWordCount(text):
        return len(nltk.word_tokenize(text))

def bzLexicalDiversity(text):
        return len(set([words.lower() for words in nltk.word_tokenize(text)])) / len(nltk.word_tokenize(text))

def bzCorrCoef(list1):
        return numpy.corrcoef(list1,[item['requester_received_pizza'] for item in data])[0,1]

def bzMean(numerator, denominator):
        return numerator / denominator


        
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
        item['SentCount']=0
        item['WordCount']=0
        item['LexicalDiversity']=0
        item['Picture']=0
        item['CurseWords']=0
        item['fWords']=0        # hypothesis that bad curse words are a turnoff
        item['CravingsTheme']=0 # hypothesis that people dont want to support your cravings
        item['SchoolTheme']=0   # hypothesis that people sympethize with hard work and trying to better oneself.
        item['AdultResponsibilityTheme']=0
        item['HealthTheme']=0   # hypothesis that people sympethize with people suffering.
        item['PayItForward']=0  # hypothesis want to help people that will one day help others.
        item['someoneFlag']=0   # a popular word in converted requests
        item['forwardFlag']=0   # a popular word in converted requests
        item['becauseFlag']=0   # a popular word in converted requests
        item['tonightFlag']=0   # a popular word in converted requests
        item['anythingFlag']=0  # a popular word in converted requests
        item['somethingFlag']=0 # a popular word in converted requests
        item['accountFlag']=0   # a popular word in converted requests
        item['throughFlag']=0   # a popular word in converted requests
        item['gettingFlag']=0   # a popular word in converted requests
        item['paycheckFlag']=0  # a popular word in converted requests

# Lexical features
for item in data:
	if len(item['request_text'])>0:
		item['SentCount']=bzSentenceCount(item['request_text'])
		item['WordCount']=bzWordCount(item['request_text'])
		item['LexicalDiversity']=bzLexicalDiversity(item['request_text'])

# Single-word flags
for item in data:
        if len(item['request_text'])>0:
                for word in nltk.word_tokenize(item['request_text']):
                        if word=='someone':
                                item['someoneFlag']+=1
                        elif word=='forward':
                                item['forwardFlag']+=1
                        elif word=='because':
                                item['becauseFlag']+=1
                        elif word=='tonight':
                                item['tonightFlag']+=1
                        elif word=='anything':
                                item['anythingFlag']+=1
                        elif word=='something':
                                item['somethingFlag']+=1
                        elif word=='account':
                                item['accountFlag']+=1
                        elif word=='through':
                                item['throughFlag']+=1
                        elif word=='getting':
                                item['gettingFlag']+=1
                        elif word=='paycheck':
                                item['paycheckFlag']+=1
                

# Pictures
picList = [item['request_id'] for item in data if re.search('.jpg|.png', item['request_text'])]

for item in data:
	if item['request_id'] in picList:
		item['Picture']=1
	else:
		item['Picture']=0

# replace with the badWords I accumulated in a private file.
#badWords = ['crap', 'poop']
badWords = ['shit', 'shitty', 'piss', 'fuck', 'fucker', 'fuckface', 'fucked', 'cunt', 'cocksucker', 'cock', 'dick', 'dickhead', 'dickless', 'tits', 'fart', 'turd', 'twat', 'asshole', 'motherfucker']
fWord = ['ass']
newbadWords = ['ass']
cravings = ['crave', 'craving', 'love', 'party', 'hangover', 'friend', 'friends', 'girlfriend', 'boyfriend', 'night', 'coming', 'late', 'starving', 'starve', 'birthday']
school = ['school', 'class', 'college', 'university', 'study', 'exam', 'exams', 'test', 'cram', 'student', 'practice', 'team', 'reading', 'read']
#schoolV2 = ['algebra', 'biology', 'geometry', 'calculus', 'chemistry', 'physics', 'geography', 'school', 'class', 'college', 'faculty', 'teacher', 'homework', 'math', 'project', 'professor', 'university', 'study', 'exam', 'exams', 'examination', 'test', 'cram', 'recite', 'speak', 'speech', 'student', 'practice', 'team', 'quiz', 'reading', 'read', 'scholar', 'undergrad', 'science', 'schoolboy', 'schoolgirl', 'tutor']
adultresponsibility = ['job', 'work', 'unemployed', 'lost', 'unemployment', 'paycheck', 'check', 'cash', 'account', 'provide', 'husband', 'wife']
health = ['doctor', 'checkup', 'medicine', 'pills', 'sad', 'depressed', 'pain']
# The following are popular words in converted text: someone,
#  forward, because, tonight, anything, something, account,
#  through, getting, paycheck


# Themes
for item in data:
        for word in nltk.word_tokenize(item['request_text']):
                if word in badWords:
                        item['CurseWords']+=1
                elif word in fWord:
                        item['fWords']+=1
                elif word in cravings:
                        item['CravingsTheme']+=1
                elif word in school:
                        item['SchoolTheme']+=1
                elif word in adultresponsibility:
                        item['HealthTheme']+=1



## Let's explore differences b/w successful requests and unsuccessful.
print('Begin calculating conversion rate...')

goodRequests = [item for item in data if item['requester_received_pizza']==1]
badRequests = [item for item in data if item['requester_received_pizza']==0]

print('...' + str((len([item for item in data if item['requester_received_pizza']==1]) / len([item for item in data])) * 100) + '% of requests are fullfilled.') 
print('')


#*******************************************************************************
# Question: Do word counts correlate with conversion?
# Answer: Somewhat
# Range: 0 - 1099
# Correlation Coefficient: Not a normal distribution.
#*******************************************************************************
print('WORD COUNTS')
print('')
print('  Range: ' + str(min([item ['WordCount'] for item in data])) + ' to ' + str(max([item ['WordCount'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['WordCount'] for item in data]), len([item ['WordCount'] for item in data]))))
if graphs == 'yes': FreqDist([item ['WordCount'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['WordCount'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['WordCount'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['WordCount'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['WordCount'] for item in data if item['requester_received_pizza']==1]), len([item ['WordCount'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['WordCount'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['WordCount'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['WordCount'] for item in data if item['requester_received_pizza']==0]), len([item ['WordCount'] for item in data if item['requester_received_pizza']==0]))))
print('')


#*******************************************************************************
# Question: Do sentence counts correlate with conversion?
# Answer: Somewhat
# Correlation Coefficient: 0.1226
#*******************************************************************************
print('SENTENCE COUNTS')
print('')
print('  Range: ' + str(min([item ['SentCount'] for item in data])) + ' to ' + str(max([item ['SentCount'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['SentCount'] for item in data]), len([item ['SentCount'] for item in data]))))
if graphs == 'yes': FreqDist([item ['SentCount'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['SentCount'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['SentCount'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['SentCount'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['SentCount'] for item in data if item['requester_received_pizza']==1]), len([item ['SentCount'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['SentCount'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['SentCount'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['SentCount'] for item in data if item['requester_received_pizza']==0]), len([item ['SentCount'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Does lexical diversity correlate with conversion?
# Answer: No
# Correlation Coefficient: -0.0651
#*******************************************************************************
print('LEXICAL DIVERSITY')
print('')
print('  Range: ' + str(min([item ['LexicalDiversity'] for item in data])) + ' to ' + str(max([item ['LexicalDiversity'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['LexicalDiversity'] for item in data]), len([item ['LexicalDiversity'] for item in data]))))
if graphs == 'yes': FreqDist([item ['LexicalDiversity'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['LexicalDiversity'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['LexicalDiversity'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['LexicalDiversity'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['LexicalDiversity'] for item in data if item['requester_received_pizza']==1]), len([item ['LexicalDiversity'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['LexicalDiversity'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['LexicalDiversity'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['LexicalDiversity'] for item in data if item['requester_received_pizza']==0]), len([item ['LexicalDiversity'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Does a cravings theme correlate with conversion?
# Answer: 
# Correlation Coefficient: 
#*******************************************************************************
print('CRAVINGS THEME')
print('')
print('  Range: ' + str(min([item ['CravingsTheme'] for item in data])) + ' to ' + str(max([item ['CravingsTheme'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['CravingsTheme'] for item in data]), len([item ['CravingsTheme'] for item in data]))))
if graphs == 'yes': FreqDist([item ['CravingsTheme'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['CravingsTheme'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['CravingsTheme'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['CravingsTheme'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['CravingsTheme'] for item in data if item['requester_received_pizza']==1]), len([item ['CravingsTheme'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['CravingsTheme'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['CravingsTheme'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['CravingsTheme'] for item in data if item['requester_received_pizza']==0]), len([item ['CravingsTheme'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Does a school theme correlate with conversion?
# Answer: Yes
# Correlation Coefficient: 0.0424
#*******************************************************************************
print('SCHOOL THEME')
print('')
print('  Range: ' + str(min([item ['SchoolTheme'] for item in data])) + ' to ' + str(max([item ['SchoolTheme'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['SchoolTheme'] for item in data]), len([item ['SchoolTheme'] for item in data]))))
if graphs == 'yes': FreqDist([item ['SchoolTheme'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['SchoolTheme'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['SchoolTheme'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['SchoolTheme'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['SchoolTheme'] for item in data if item['requester_received_pizza']==1]), len([item ['SchoolTheme'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['SchoolTheme'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['SchoolTheme'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['SchoolTheme'] for item in data if item['requester_received_pizza']==0]), len([item ['SchoolTheme'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Do picture attachments correlate with conversion?
# Answer: 
#*******************************************************************************
print('PICTURES')
print('')
print('  Range: ' + str(min([item ['Picture'] for item in data])) + ' to ' + str(max([item ['Picture'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['Picture'] for item in data]), len([item ['Picture'] for item in data]))))
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['Picture'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['Picture'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['Picture'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['Picture'] for item in data if item['requester_received_pizza']==1]), len([item ['Picture'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['Picture'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['Picture'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['Picture'] for item in data if item['requester_received_pizza']==0]), len([item ['Picture'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Are curse words a turnoff?
# Answer: No, but I believe we should change this to curse word counts or
#      severity scores.
# Correlation Coefficient: -0.0141
#*******************************************************************************
print('CURSE WORDS')
print('')
print('  Range: ' + str(min([item ['CurseWords'] for item in data])) + ' to ' + str(max([item ['CurseWords'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['CurseWords'] for item in data]), len([item ['CurseWords'] for item in data]))))
if graphs == 'yes': FreqDist([item ['CurseWords'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['CurseWords'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['CurseWords'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['CurseWords'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['CurseWords'] for item in data if item['requester_received_pizza']==1]), len([item ['CurseWords'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['CurseWords'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['CurseWords'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['CurseWords'] for item in data if item['requester_received_pizza']==0]), len([item ['CurseWords'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Is the "F" word a turnoff?
# Answer: 
# Correlation Coefficient: 
#*******************************************************************************
print('F-WORD')
print('')
print('  Range: ' + str(min([item ['fWords'] for item in data])) + ' to ' + str(max([item ['fWords'] for item in data])))
print('  Mean: ' + str(bzMean(sum([item ['fWords'] for item in data]), len([item ['fWords'] for item in data]))))
if graphs == 'yes': FreqDist([item ['fWord'] for item in data]).plot()
print('  Correlation Coefficient: ' + str(bzCorrCoef([item['fWords'] for item in data])))
print('')
print('  Converted requests')
print('    Range: ' + str(min([item['fWords'] for item in data if item['requester_received_pizza']==1])) + ' to ' + str(max([item['fWords'] for item in data if item['requester_received_pizza']==1])))
print('    Mean: ' + str(bzMean(sum([item ['fWords'] for item in data if item['requester_received_pizza']==1]), len([item ['fWords'] for item in data if item['requester_received_pizza']==1]))))
print('')
print('  Unconverted requests')
print('    Range: ' + str(min([item['fWords'] for item in data if item['requester_received_pizza']==0])) + ' to ' + str(max([item['fWords'] for item in data if item['requester_received_pizza']==0])))
print('    Mean: ' + str(bzMean(sum([item ['fWords'] for item in data if item['requester_received_pizza']==0]), len([item ['fWords'] for item in data if item['requester_received_pizza']==0]))))
print('')



#*******************************************************************************
# Question: Are there differences between word-length frequencies of converted
#      vs. unconverted requests?
# Answer: No
# Correlation Coefficient: 
#*******************************************************************************
print('Begin calculating word length frequencies...')
      
cnvtText = ' '.join([item['request_text'] for item in data
                     if len(item['request_text'])>0
                     and item['requester_received_pizza']==1])
wl1 = [len(word) for word in nltk.word_tokenize(cnvtText) if word.isalpha()]
wl1fd = FreqDist(wl1)
if graphs == 'yes': wl1fd.plot()
## 4, 3, 2, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 18
print('...Word length frequencies for successful requests have been plotted.')

uncnvtText = ' '.join([item['request_text'] for item in data
                     if len(item['request_text'])>0
                     and item['requester_received_pizza']==0])
wl2 = [len(word) for word in nltk.word_tokenize(uncnvtText) if word.isalpha()]
wl2fd = FreqDist(wl2)
if graphs == 'yes': wl2fd.plot()
## 4, 3, 2, 5, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 17, 35, 20
print('...Word length frequencies for unsuccessful requests have been plotted.')



#*******************************************************************************
# Question: Do requests without text ever get converted?
# Answer: Yes, >17%
# Correlation Coefficient: See word or sentence count corr coef
#*******************************************************************************
print('Do requests without text ever get converted?')

noTextCount = len([item['requester_received_pizza'] for item in data if len(item['request_text'])==0])
noTextSum = sum([item['requester_received_pizza'] for item in data if len(item['request_text'])==0])

print(noTextSum / noTextCount)



#*******************************************************************************
# Question: Do people with multiple requests have better success?
# Answer:
# Correlation Coefficient:
#*******************************************************************************
users = {}
for item in data:
        users[item['request_username']] = users.get(item['request_username'],0)+1




#*******************************************************************************
# Question: Are there 5 words used frequently in the uncoverted text that are
#      not used at all in the converted text?
# Answer:
# Correlation Coefficient:
#*******************************************************************************

someone
forward
because
tonight
anything
something
account
through
getting
paycheck


## Let's evaluate frequently occuring, large words across the two datasets.

print('Begin important word search...')

FreqDist([word.lower() for word in nltk.word_tokenize(cnvtText) if word.isalpha() and len(word) > 6]).plot(50)
FreqDist([word.lower() for word in nltk.word_tokenize(uncnvtText) if word.isalpha() and len(word) > 6]).plot(50)

## I see some patterns around cravings, school and employment.
## I should probably explore keywords around these themes.
## CRAVINGS - crave, craving, love, party, hangover, friend, friends, girlfriend, boyfriend, night, coming, late, starving, starve, birthday
## SCHOOL - school, class, college, university, study, exam, exams, test, cram, student, practice, team, reading, read
## EMPLOYMENT - job, work, unemployed, lost, unemployment, paycheck, check, cash, account, provide, husband, wife
## HEALTH - doctor, checkup, medicine, pills, sad, depressed, pain

## I also see themes around gratitude and politeness
## appreciate, thanks, greatly, grateful, thank


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
