from TweetDatasetBert2 import VectorizedTweetsDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from ToTensor import ToTensor

#%% data loader generating --> train dataset

#transformed_dataset = VectorizedTweetsDataset('training.cleaned.csv',
#                                              transform=transforms.Compose([ToTensor()]))
#
#dataloader = DataLoader(transformed_dataset, batch_size=4,
#                        shuffle=False, num_workers=0)
#
#for i_batch, sample_batched in enumerate(dataloader):
#    print(i_batch, sample_batched['tweet'],
#          sample_batched['polarity'])
    
#%% data loader generating --> test dataset
#transformed_dataset = VectorizedTweetsDataset('testing.cleaned.csv',
#                                              transform=transforms.Compose([ToTensor()]))
#
#dataloader = DataLoader(transformed_dataset, batch_size=4,
#                        shuffle=False, num_workers=0)
#
#for i_batch, sample_batched in enumerate(dataloader):
#    print(i_batch, sample_batched['tweet'],
#          sample_batched['polarity'])

#%% Bert dataset test
sentence_length = 300
transformed_dataset = VectorizedTweetsDataset('training.cleaned.csv', sentence_length)

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=False, num_workers=0)
c = 0
for i_batch, sample_batched in enumerate(dataloader):
    c += 1
    print(sample_batched['tweet'].size())
    print(sample_batched['attention_mask'].size())
    print(sample_batched['polarity'].size())
    if c > 5: break