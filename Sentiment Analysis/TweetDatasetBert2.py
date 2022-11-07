#%% Importing 
from __future__ import print_function, division
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer

#%% Class Defenition
class VectorizedTweetsDataset(Dataset):

    def __init__(self, csv_file, sen_len):

        self.tweets = pd.read_csv(csv_file, encoding="ISO-8859-1")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sen_len = sen_len
        
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        polarity = self.tweets.iloc[idx, 0]     
        tweet = self.tweets.iloc[idx, 1]
        
        encoded_dict = self.tokenizer.encode_plus(
                    tweet,                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = 300,           # Pad & truncate all sentences.
                    pad_to_max_length = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt')     # Return pytorch tensors
    
        input_ids = encoded_dict['input_ids']
        attn_mask = encoded_dict['attention_mask']
        labels = torch.tensor(polarity)
#        attn_mask = (attention_masks != 0).long()
        sample = {'tweet': input_ids, 'attention_mask': attn_mask ,'polarity': labels}
   
        return sample