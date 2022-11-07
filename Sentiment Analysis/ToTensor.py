import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, tweet):
        return {'tweet': torch.from_numpy(tweet['tweet']).type(torch.FloatTensor),
                'polarity': tweet['polarity']}