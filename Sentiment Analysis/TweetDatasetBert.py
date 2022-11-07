#%% Importing 
from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

#%% Class Defenition
class VectorizedTweetsDataset(Dataset):

    def __init__(self, csv_file, sen_len, transform=None):

        self.tweets = pd.read_csv(csv_file, encoding="ISO-8859-1")
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sen_len = sen_len
        
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        polarity = self.tweets.iloc[idx, 0]
        
        tweet = self.tweets.iloc[idx, 1]
        vocabs = tweet.split()
    
        vectors = []
        for vocab in vocabs:
            vectors.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(vocab)))
            
        if len(vectors)<self.sen_len:
            for n in range(self.sen_len-len(vectors)):
                vectors.append(0)
        elif len(vectors)>self.sen_len:
             for n in range(len(vectors-self.sen_len)):
                vectors.pop()
                
        sample = {'tweet': np.array(vectors), 'polarity': polarity}

        if self.transform:
            sample = self.transform(sample)
            
        return sample