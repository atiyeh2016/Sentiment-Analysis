from Preprocessing import Preprocessing
import csv

#%% data preprocessing --> train

#P = Preprocessing()
#with open('training.1600000.processed.noemoticon.csv') as csvfile:
#    dirtytweets = csv.reader(csvfile)
#    for dirtytweet in dirtytweets:
#        polarity = dirtytweet[0]
#        tweet = dirtytweet[5]
#        P.process_tweet(tweet, polarity)
#       
##%% preprocessing and making new csv --> train
#
#with open('training.cleaned.csv', 'w', newline='') as csvfile:
#    
#    fieldnames = ['polarity','text']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    writer.writeheader()
#    
#    for cleaned_text, new_polarity in zip(P.tweet, P.label):
#        cleaned_tweet = {'polarity': new_polarity ,'text': cleaned_text}
#        writer.writerow(cleaned_tweet)

#%% data preprocessing --> test
        
P = Preprocessing()
with open('testdata.manual.2009.06.14.csv') as csvfile:
    dirtytweets = csv.reader(csvfile)
    for dirtytweet in dirtytweets:
        polarity = dirtytweet[0]
        tweet = dirtytweet[5]
        P.process_tweet(tweet, polarity)

#%% preprocessing and making new csv --> test

with open('testing.cleaned.csv', 'w', newline='') as csvfile:
    
    fieldnames = ['polarity','text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for cleaned_text, new_polarity in zip(P.tweet, P.label):
        cleaned_tweet = {'polarity': new_polarity ,'text': cleaned_text}
        writer.writerow(cleaned_tweet)
