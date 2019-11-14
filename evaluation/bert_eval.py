import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel
from torch.utils.data import DataLoader
from data_loader import ActivityNetCaptionDataset
import numpy as np

import sys
import pickle

GPU = True

def batchPadding(batch):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    batch_size = len(batch)
    tokenizedSentences = []

    labels = []
    mask_positions = []

    max_len = 0
    for i in range(batch_size):
        data = batch[i]
        indexed_tokens = data[0]
        segments_ids = data[1]
        labels.append(data[2])
        mask_positions.append(data[3])
        total_len = len(indexed_tokens)
        if total_len > max_len:
            max_len = total_len
        tokenizedSentences.append((torch.LongTensor(indexed_tokens), torch.LongTensor(segments_ids)))
    
    batch_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    segments_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    

    for i in range(len(tokenizedSentences)):
        pair = tokenizedSentences[i]
        len_pair = len(pair[0])
        batch_tensor[i, :len_pair] = pair[0]
        segments_tensor[i, :len_pair] = pair[1]
        attention_mask[i, :len_pair] = 1

    return (batch_tensor, segments_tensor, attention_mask, labels, mask_positions)

    

def loadData(data_path, batch_size):
    dataset = ActivityNetCaptionDataset(data_path)
    eval_dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=batchPadding)

    total_train_data = len(dataset)
    indices = list(range(total_train_data))
    np.random.shuffle(indices)
    split = int(total_train_data * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    finetune_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, collate_fn=batchPadding)
    finetune_val = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler, collate_fn=batchPadding)
    
    return eval_dataLoader, finetune_train, finetune_val


def train(data, max_epoch, model, optimizer, PATH):
    
    model.train()
    running_loss = 0
    for epoch in range(max_epoch):

        for n, batch in enumerate(data):
            optimizer.zero_grad()
            batch_tensor, segments_tensor, attention_mask, labels, mask_positions = batch
            if GPU:
                batch_tensor = batch_tensor.cuda()
                segments_tensor = segments_tensor.cuda()
                attention_mask = attention_mask.cuda()             
            
            output = model(batch_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask, masked_lm_labels=batch_tensor)
            loss, score = output
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n%50 == 0 and n != 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, running_loss/50))
                running_loss = 0
        model.save_pretrained(PATH)
    return

def eval(data, model):
    model.eval()
    for batch in data:
        batch_tensor, segments_tensor, attention_mask, labels, mask_positions = batch
        if GPU:
            batch_tensor = batch_tensor.cuda()
            segments_tensor = segments_tensor.cuda()
            attention_mask = attention_mask.cuda()             
        
        output = model(batch_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask, masked_lm_labels=batch_tensor)
        score = output[1]
        predicted_index = torch.argmax(score[mask_positions])

def main():
    lr = 0.00005
    PATH = '/home/ruoyaow/toys/bert/'

    if len(sys.argv) > 1 and sys.argv[1] == 'c':
        pretrained = PATH
    else:
        pretrained = 'bert-base-uncased'

    model = BertForMaskedLM.from_pretrained(pretrained, output_hidden_states=True, output_attentions=False)
    if GPU:
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr)
    max_epoch = 10
    batch_size = 32

    train_data = 'noun_blank.txt'
    evaluation, train, test  = loadData(train_data, batch_size)
    
    eval(evaluation, model)

    train(train, max_epoch, model, optimizer, PATH)
    eval(test, model)


if __name__ == "__main__":
    main()
