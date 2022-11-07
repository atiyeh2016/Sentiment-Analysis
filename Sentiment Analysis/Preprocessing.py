import re

class Preprocessing:
    
    def __init__(self):
        self.tweet = []
        self.label = []
        
    def eliminate_link(self, tweet):
        return 1 if not re.match('(.*?)http.*?\s?(.*?)', tweet) else 0
    
    def remove_hashtag(self, tweet):
        if "#" in tweet:
            vocabs = tweet.split()
            for i, vocab in enumerate(vocabs):
                if "#" in vocab:
                    vocabs[i] = "HASHTAG"
            return " ".join(vocabs)
        else:
            return tweet
        
    def remove_mention(self, tweet):
        if "@" in tweet:
            vocabs = tweet.split()
            for i, vocab in enumerate(vocabs):
                if "@" in vocab:
                    vocabs[i] = "MENTION"
            return " ".join(vocabs)
        else:
            return tweet
        
    def remove_punctuation(self, tweet):
        return re.sub(r'[^\w\s]','',tweet)
    
    def change_label(self, label):
        if label == '0':
            return '0'
        elif label == '2':
            return '1'
        elif label == '4':
            return '2'
        
    def process_tweet(self, tweet, label):
        flag = self.eliminate_link(tweet)
        if flag:
            tweet = self.remove_hashtag(tweet)
            tweet = self.remove_mention(tweet)
            tweet = self.remove_punctuation(tweet)
            self.tweet.append(tweet)
            
            label = self.change_label(label)
            self.label.append(label)

        
        