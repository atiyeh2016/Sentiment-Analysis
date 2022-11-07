#%% Importing 
from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import pickle

a_file = open("vectorized_words.pkl", "rb")
embeddings_dict = pickle.load(a_file)
a_file.close()

b_file = open("mean_of_vectors.pkl", "rb")
mean_v, zero_vector = pickle.load(b_file)
b_file.close()

#%% Class Defenition
class VectorizedTweetsDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.tweets = pd.read_csv(csv_file, encoding="ISO-8859-1")
        self.transform = transform
        
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
            try:
                vectors.append(embeddings_dict[vocab.lower()])
            except:
                vectors.append(mean_v)
            
        for n in range(280-len(vocabs)):
            vectors.append(zero_vector)
                
        sample = {'tweet': np.array(vectors), 'polarity': polarity}

        if self.transform:
            sample = self.transform(sample)
            

        return sample