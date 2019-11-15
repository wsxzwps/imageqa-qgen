import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader
from data_loader import ActivityNetCaptionDataset
from torch.utils.data.sampler import SubsetRandomSampler
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
        masked_lm_label = data[4]
        labels.append(data[2])
        mask_positions.append(data[3])
        total_len = len(indexed_tokens)
        if total_len > max_len:
            max_len = total_len
        tokenizedSentences.append((torch.LongTensor(indexed_tokens), torch.LongTensor(segments_ids), torch.LongTensor(masked_lm_label)))
    
    batch_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    segments_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    masked_lm_labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    

    for i in range(len(tokenizedSentences)):
        pair = tokenizedSentences[i]
        len_pair = len(pair[0])
        batch_tensor[i, :len_pair] = pair[0]
        segments_tensor[i, :len_pair] = pair[1]
        attention_mask[i, :len_pair] = 1
        mask_positions[i, :len_pair] = pair[2]

    return (batch_tensor, segments_tensor, attention_mask, labels, mask_positions, masked_lm_labels)

    

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
            batch_tensor, segments_tensor, attention_mask, labels, mask_positions, masked_lm_labels = batch
            if GPU:
                batch_tensor = batch_tensor.cuda()
                segments_tensor = segments_tensor.cuda()
                attention_mask = attention_mask.cuda()             
            
            output = model(batch_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)
            loss = output[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if n%50 == 0 and n != 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, running_loss/50))
                running_loss = 0
        model.save_pretrained(PATH)
    return model

def eval(data, model, tokenizer):
    model.eval()
    correct = 0
    total_num = 0
    for batch in data:
        batch_tensor, segments_tensor, attention_mask, labels, mask_positions, masked_lm_labels = batch
        if GPU:
            batch_tensor = batch_tensor.cuda()
            segments_tensor = segments_tensor.cuda()
            attention_mask = attention_mask.cuda()             
        batch_size = batch_tensor.shape[0]
        output = model(batch_tensor, token_type_ids=segments_tensor, attention_mask=attention_mask, masked_lm_labels=masked_lm_labels)
        score = output[1]
        predicted_index = torch.argmax(score[list(range(batch_size)), mask_positions], dim=1)
        out_text = tokenizer.convert_ids_to_tokens(predicted_index.tolist())
        total_num += batch_size
        for i in range(batch_size):
            if labels[i] == out_text[i]:
                correct += 1
    acc = correct / total_num
    print(acc)

def main():
    lr = 0.00005
    PATH = '/home/ruoyaow/imageqa-qgen/evaluation'

    if len(sys.argv) > 1 and sys.argv[1] == 'c':
        pretrained = PATH
    else:
        pretrained = 'bert-base-uncased'

    model = BertForMaskedLM.from_pretrained(pretrained, output_hidden_states=True, output_attentions=False)
    if GPU:
        model = model.cuda()

    optimizer = AdamW(model.parameters(), lr=lr)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    max_epoch = 1
    batch_size = 32

    train_data = 'noun_blank.txt'
    evaluation, trainld, testld  = loadData(train_data, batch_size)
    
    # eval(evaluation, model, tokenizer)

    # model = train(trainld, max_epoch, model, optimizer, PATH)
    eval(trainld, model, tokenizer)
    eval(testld, model, tokenizer)


if __name__ == "__main__":
    main()
