import torch
from torch.utils.data import Dataset
import numpy as np

from utils import get_timestamp


class SuicidalDataset(Dataset):
    def __init__(self, label, tweet, temporal, timestamp, current=True, random=False):
        super().__init__()
        self.label = label
        self.tweet = tweet
        self.temporal = temporal
        self.current = current
        self.timestamp = timestamp
        self.random = random

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        labels = torch.tensor(self.label[item])
        tweet_features = self.tweet[item]
        if self.current:
            result = self.temporal[item]
            if self.random:
                np.random.shuffle(result)
            temporal_tweet_features = torch.tensor(result)
            timestamp = torch.tensor(get_timestamp(self.timestamp[item]))
        else:
            if len(self.temporal[item]) == 1:
                temporal_tweet_features = torch.zeros((1, 768), dtype=torch.float32)
                timestamp = torch.zeros((1, 1), dtype=torch.float32)
            else:
                temporal_tweet_features = torch.tensor(self.temporal[item][1:])
                timestamp = torch.tensor(get_timestamp(self.timestamp[item][1:]))

        return [labels, tweet_features, temporal_tweet_features, timestamp]
