from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModel, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
import torch
import numpy as np
import time


def extract_embedding(df, device, model='gpt', max_len = 100):
    output_path = 'out/'+model+'_'+str(len(df))+'.pt'
    print(output_path)
    if model == 'gpt':
        print('Model vertified: GPT')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='0')
        model = GPT2Model.from_pretrained('gpt2').to(device)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 100)
    elif model == 'codegpt':
        print('Model vertified: CodeGPT')
        tokenizer = GPT2Tokenizer.from_pretrained('AISE-TUDelft/CodeGPT-Multilingual', pad_token='0')
        model = GPT2Model.from_pretrained('AISE-TUDelft/CodeGPT-Multilingual').to(device)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 100)
    elif model == 'codebert':
        print('Model vertified: CodeBERT') 
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 100)
    elif model == 'bert':
        print('Model vertified: BERT')
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("google-bert/bert-base-uncased").to(device)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 100)
    elif model == 'roberta':
        print('Model vertified: RoBERTa')
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base").to(device)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 100)
    elif model == 'miroslaw':
        print('Model vertified: Miroslaw')
        tokenizer = AutoTokenizer.from_pretrained('mstaron/SingletonBERT')
        model = AutoModelForMaskedLM.from_pretrained("mstaron/SingletonBERT", output_hidden_states=True).to(device)
        embed_from_local(df, device, output_path, tokenizer, model, max_len = 100)
    elif model == 'epoch4_roberta':
        print('Model vertified:', 'epoch4_roberta')
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_length=512)
        model = RobertaForMaskedLM.from_pretrained('model/epoch4_roberta', output_hidden_states=True).to(device)
        embed_from_local(df, device, output_path, tokenizer, model, max_len = 100)
    else:
        print('Model vertified: saved model')
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        model = RobertaForMaskedLM.from_pretrained('model/'+model, output_hidden_states=True).to(device)
        embed_from_local(df, device, output_path, tokenizer, model, max_len = 100)
    
    return output_path



def embed_from_model(df, device, output_path, tokenizer, model, max_len = 100):

    time_start = time.time()
    print('Starting extract embeddings......')
    for i in range(len(df)):
        oneCode = df['code'].values[i]
        splitCode = oneCode.splitlines()
        splitCode = list(filter(None, splitCode))
        sum_embedding = torch.zeros(1,768).to(device)
        for codeLine in splitCode:
            encoded_input = tokenizer(codeLine,
                                    max_length=max_len,
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors='pt').to(device)
            line_embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()
            sum_embedding = torch.add(sum_embedding,line_embeddings)
        avg_embedding = sum_embedding/len(splitCode)
        if i == 0 :
            embeddings = avg_embedding.detach()
        else:
            embeddings = torch.cat((embeddings, avg_embedding.detach()),0)
        if (i+1) % 10000 == 0:
            print('Time elapsed: {:.2f} seconds, Data processed:{}'.format(time.time()-time_start, i+1))
            time_start = time.time()
    torch.save(embeddings, output_path)
    print('End of extracting...Number of record:', str(embeddings.shape))



    # line_embeddings=model(torch.tensor(line_tokens_ids)[None,:].to(device)).pooler_output ## last_hidden_state size: torch.Size([1, 200, 768]), pooler size: torch.Size([1, 768])
    # sum_embedding = torch.add(sum_embedding, line_embeddings)

def embed_from_local(df, device, output_path, tokenizer, model, max_len = 100):

    time_start = time.time()
    print('Starting extract embeddings......')
    for i in range(len(df)):
        oneCode = df['code'].values[i]
        splitCode = oneCode.splitlines()
        splitCode = list(filter(None, splitCode))
        sum_embedding = torch.zeros(1,768).to(device)
        for codeLine in splitCode:
            encoded_input = tokenizer(codeLine,
                                    max_length=max_len,
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors='pt').to(device)
            line_embeddings = model(**encoded_input).hidden_states[12].mean(dim=1).detach()
            sum_embedding = torch.add(sum_embedding,line_embeddings)
        avg_embedding = sum_embedding/len(splitCode)
        if i == 0 :
            embeddings = avg_embedding.detach()
        else:
            embeddings = torch.cat((embeddings, avg_embedding.detach()),0)
        if (i+1) % 10000 == 0:
            print('Time elapsed: {:.2f} seconds, Data processed:{}'.format(time.time()-time_start, i+1))
            time_start = time.time()
    torch.save(embeddings, output_path)
    print('End of extracting...Number of record:', str(embeddings.shape))

    # tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_length=512)
    # model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)