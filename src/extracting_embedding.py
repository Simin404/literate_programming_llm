from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModel, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, AutoModelForCausalLM
import torch
import time

def extract_embedding(df, device, model='gpt'):
    output_path = 'out/'+model+'_'+str(len(df))+'.pt'
    access_token = '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'
    print(output_path)
    if model == 'gpt':
        print('Model vertified: GPT')
        model_id = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_id, pad_token='0')
        model = GPT2Model.from_pretrained(model_id).to(device)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 1024)
    elif model == 'codegpt':
        print('Model vertified: CodeGPT')
        model_id = 'AISE-TUDelft/CodeGPT-Multilingual'
        tokenizer = GPT2Tokenizer.from_pretrained(model_id, pad_token='0')
        model = GPT2Model.from_pretrained(model_id).to(device)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 1024)
    elif model == 'codebert':
        print('Model vertified: CodeBERT') 
        model_id = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 512)
    elif model == 'bert':
        print('Model vertified: BERT')
        model_id = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 512)
    elif model == 'roberta':
        print('Model vertified: RoBERTa')
        model_id = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_id)
        model = RobertaModel.from_pretrained(model_id).to(device)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 512)
    elif model == 'falcon7b':
        print('Model vertified: falcon7b')
        model_id = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 2048)
    elif model == 'falcon11b':
        print('Model vertified: falcon11b')
        model_id = "tiiuae/falcon-11b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 4096)
    elif model == 'falcon40b':
        print(device)
        print_available_gpus_with_memory()
        print('Model vertified: falcon40b')
        model_id = "tiiuae/falcon-40b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 2048)

    elif model == 'llama7b':
        print('Model vertified: llama7b')
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 4096)
    elif model == 'llama8b':
        print('Model vertified: llama8b')
        model_id = "meta-llama/Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 4096)
    elif model == 'llama13b':
        print('Model vertified: llama13b')
        model_id = "meta-llama/Llama-2-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 4096)
    elif model == 'llama70b':
        print('Model vertified: llama70b')
        model_id = "meta-llama/Llama-2-70b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        print('Done loading model:', model_id)
        embed_from_model(df, device, output_path, tokenizer, model, max_len = 4096)
    else:
        print('Model vertified: saved model')
        output_path = '../out/'+model+'_'+str(len(df))+'.pt'
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_length=512)
        model = RobertaForMaskedLM.from_pretrained('../model/'+ model, output_hidden_states=True).to(device)
        print('Done loading model:', model_id)
        embed_from_local(df, device, output_path, tokenizer, model, max_len = 100)
    
    return output_path

def print_available_gpus_with_memory():
    if torch.cuda.is_available():
        print("Available GPUs and their memory information:")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert bytes to GB
            print(f"GPU {i}: {device_name}, Total Memory: {total_memory:.2f} GB")
    else:
        print("No GPUs available.")


def embed_from_split(df, device, output_path, tokenizer, model, max_len = 100):
    model.eval()
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
                                    padding=False,
                                    return_tensors='pt').to(device)
            with torch.no_grad():
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

def embed_from_model(df, device, output_path, tokenizer, model, max_len=2048):
    model.eval()
    time_start = time.time()
    print("Starting to extract embeddings...")
    all_embeddings = []
    for i in range(len(df)):
        text = df['code'].values[i]
    
        encoded_input = tokenizer(
            text,  # Only one input at a time
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()

        all_embeddings.append(embeddings.cpu())

        # Log progress for every 5000 processed samples
        if (i + 1) % 5000 == 0:
            print(f"Time elapsed: {time.time() - time_start:.2f} seconds, Data processed: {i + 1}")
            time_start = time.time()
 
    # Save the embeddings to the output path
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, output_path)
    print("End of extraction. Number of records:", str(all_embeddings.shape))


def embed_from_local(df, device, output_path, tokenizer, model, max_len = 100):
    model.eval()
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
            ## Different from downloaded model
            with torch.no_grad():
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


