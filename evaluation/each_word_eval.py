import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader
from data_loader import ActivityNetCaptionDataset
from torch.utils.data.sampler import SubsetRandomSampler
from warmup_scheduler import GradualWarmupScheduler
import numpy as np

import sys
import pickle

GPU = True

with open('../train.txt', 'r') as f:
    data = f.readlines()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=False)

word_dict = {}

