from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModel, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, AutoModelForCausalLM
import torch
import time

def combined_embedding(train_df, test_df, desc_df, vis_df, device, model='gpt'):
    train_path = 'out/train/'+model+'_'+str(len(train_df))+'.pt'
    test_path = 'out/test/'+model+'_'+str(len(test_df))+'.pt'
    desc_path = 'out/desc/'+model+'_'+str(len(desc_df))+'.pt'
    vis_path = 'out/visualization/'+model+'_'+str(len(vis_df))+'.pt'
    access_token = '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'
    print(train_path, test_path, desc_path, vis_path)
    if model == 'code7b':
        print('Model vertified: code7b')
        model_id = "codellama/CodeLlama-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'code13b':
        print('Model vertified: code13b')
        model_id = "codellama/CodeLlama-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'code34b':
        print('Model vertified: code34b')
        model_id = "codellama/CodeLlama-34b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'code70b':
        print('Model vertified: code70b')
        model_id = "codellama/CodeLlama-70b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'falcon11b':
        print('Model vertified: falcon11b')
        model_id = "tiiuae/falcon-11b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'falcon7b':
        print('Model vertified: falcon7b')
        model_id = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'falcon40b':
        print('Model vertified: falcon40b')
        model_id = "tiiuae/falcon-40b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama8b':
        print('Model vertified: llama8b')
        model_id = "meta-llama/Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama70b':
        print('Model vertified: llama70b')
        model_id = "meta-llama/Llama-3.1-70B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama7b':
        print('Model vertified: llama7b')
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama13b':
        print('Model vertified: llama13b')
        model_id = "meta-llama/Llama-2-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'bert':
        print('Model vertified: BERT')
        model_id = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'roberta':
        print('Model vertified: RoBERTa')
        model_id = "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'gpt':
        print('Model vertified: GPT')
        model_id = 'gpt2'
        tokenizer = GPT2Tokenizer.from_pretrained(model_id, pad_token='0')
        model = GPT2Model.from_pretrained(model_id).to(device)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'codegpt':
        print('Model vertified: CodeGPT')
        model_id = 'AISE-TUDelft/CodeGPT-Multilingual'
        tokenizer = GPT2Tokenizer.from_pretrained(model_id, pad_token='0')
        model = GPT2Model.from_pretrained(model_id).to(device)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'codebert':
        print('Model vertified: CodeBERT') 
        model_id = "microsoft/codebert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).to(device)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    else:
        print('wrong model name')
    

def embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_len):
    model.eval()
    time_start = time.time()
    print("Starting to extract embeddings...")
    train_embeddings = []
    test_embeddings = []
    desc_embeddings = []
    vis_embeddings = []
    for i in range(len(train_df)):
        text = train_df['code'].values[i]
        encoded_input = tokenizer(
            text,  # Only one input at a time
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()
        train_embeddings.append(embeddings.cpu())
        if (i + 1) % 5000 == 0:
            print(f"Time elapsed: {time.time() - time_start:.2f} seconds, Data processed: {i + 1}")
            time_start = time.time()
    # Save the embeddings to the output path
    train_embeddings = torch.cat(train_embeddings, dim=0)
    torch.save(train_embeddings, train_path)
    print("End of extraction. Number of records:", str(train_embeddings.shape))

    for i in range(len(test_df)):
        text = test_df['code'].values[i]
        encoded_input = tokenizer(
            text,  # Only one input at a time
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()
        test_embeddings.append(embeddings.cpu())
        if (i + 1) % 5000 == 0:
            print(f"Time elapsed: {time.time() - time_start:.2f} seconds, Data processed: {i + 1}")
            time_start = time.time()
    # Save the embeddings to the output path
    test_embeddings = torch.cat(test_embeddings, dim=0)
    torch.save(test_embeddings, test_path)
    print("End of extraction. Number of records:", str(test_embeddings.shape))

    desc_lists = desc_df['task_description']
    for i in range(len(desc_lists)):
        text = desc_lists[i]
        encoded_input = tokenizer(
            text,  # Only one input at a time
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()
        desc_embeddings.append(embeddings.cpu())
        if (i + 1) % 5000 == 0:
            print(f"Time elapsed: {time.time() - time_start:.2f} seconds, Data processed: {i + 1}")
            time_start = time.time()
    # Save the embeddings to the output path
    desc_embeddings = torch.cat(desc_embeddings, dim=0)
    torch.save(desc_embeddings, desc_path)
    print("End of extraction. Number of records:", str(desc_embeddings.shape))

    for i in range(len(vis_df)):
        text = vis_df['code'].values[i]
        encoded_input = tokenizer(
            text,  # Only one input at a time
            max_length=max_len,
            truncation=True,
            padding=False,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()
        vis_embeddings.append(embeddings.cpu())
        if (i + 1) % 5000 == 0:
            print(f"Time elapsed: {time.time() - time_start:.2f} seconds, Data processed: {i + 1}")
            time_start = time.time()
    # Save the embeddings to the output path
    vis_embeddings = torch.cat(vis_embeddings, dim=0)
    torch.save(vis_embeddings, vis_path)
    print("End of extraction. Number of records:", str(vis_embeddings.shape))

    








