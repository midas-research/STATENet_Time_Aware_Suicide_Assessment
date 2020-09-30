import datetime

import numpy as np
import torch
import torch.nn as nn
import transformers
#from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def pad_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens]


def pad_ts_collate(batch):
    target = [item[0] for item in batch]
    tweet = [item[1] for item in batch]
    data = [item[2] for item in batch]
    timestamp = [item[3] for item in batch]

    lens = [len(x) for x in data]

    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    timestamp = nn.utils.rnn.pad_sequence(timestamp, batch_first=True, padding_value=0)

    #     data = torch.tensor(data)
    target = torch.tensor(target)
    tweet = torch.tensor(tweet)
    lens = torch.tensor(lens)

    return [target, tweet, data, lens, timestamp]


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append(datetime.datetime.timestamp(t))

    np.array(timestamp) - timestamp[-1]
    return timestamp
