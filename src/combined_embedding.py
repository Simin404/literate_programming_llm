from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModel, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, AutoModelForCausalLM
import torch
import time
import os
from dotenv import load_dotenv
import argparse
import pandas as pd

def combined_embedding(train_df, test_df, desc_df, vis_df=None, device=None, model='gpt'):
    train_path = 'out/train/'+model+'_'+str(len(train_df))+'.pt'
    test_path = 'out/test/'+model+'_'+str(len(test_df))+'.pt'
    desc_path = 'out/desc/'+model+'_'+str(len(desc_df))+'.pt'
    if vis_df is not None:
        vis_path = 'out/visualization/'+model+'_'+str(len(vis_df))+'.pt'
    else:
        vis_path = ''

    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_API_KEY")

    print(train_path, test_path, desc_path, vis_path)
    if model == 'code7b':
        print('Model vertified: code7b')
        model_id = "codellama/CodeLlama-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        print('Done loading tokenizer')
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'code13b':
        print('Model vertified: code13b')
        model_id = "codellama/CodeLlama-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        print('Done loading tokenizer')
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'code34b':
        print('Model vertified: code34b')
        model_id = "codellama/CodeLlama-34b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        print('Done loading tokenizer')
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'code70b':
        print('Model vertified: code70b')
        model_id = "codellama/CodeLlama-70b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        print('Done loading tokenizer')
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'falcon11b':
        print('Model vertified: falcon11b')
        model_id = "tiiuae/falcon-11b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'falcon7b':
        print('Model vertified: falcon7b')
        model_id = "tiiuae/falcon-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'falcon40b':
        print('Model vertified: falcon40b')
        model_id = "tiiuae/falcon-40b"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, use_flash_attention_2=False)
        max_length = tokenizer.model_max_length
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama8b':
        print('Model vertified: llama8b')
        model_id = "meta-llama/Llama-3.1-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama70b':
        print('Model vertified: llama70b')
        model_id = "meta-llama/Llama-3.1-70B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama7b':
        print('Model vertified: llama7b')
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
        max_length = model.config.max_position_embeddings
        print('Done loading model:', model_id, max_length)
        embed_from_model(train_df, test_df, desc_df, vis_df, device, train_path, test_path, desc_path, vis_path, tokenizer, model, max_length)
    elif model == 'llama13b':
        print('Model vertified: llama13b')
        model_id = "meta-llama/Llama-2-13b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto",trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True, token=access_token, use_flash_attention_2=False)
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
    if not os.path.exists(train_path):
        chunk_idx = 0
        start_idx = 0
        end_idx =   len(train_df)
        save_steps = 100000
        base_path = train_path.with_suffix('')
        while os.path.exists(f"{base_path}_part{chunk_idx}{train_path.suffix}"):
            chunk_idx += 1
            start_idx += save_steps
        print(f"Start point: {start_idx}")
        train_embeddings = []
        print(f"Start point: {start_idx}, End point: {end_idx}")
        for i in range(start_idx, end_idx):
            text = train_df['code'].values[i]
            encoded_input = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding=False,
                return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                embeddings = model(**encoded_input).last_hidden_state.mean(dim=1).detach()
            train_embeddings.append(embeddings.cpu())

            if (i + 1) % save_steps == 0:
                chunk_tensor = torch.cat(train_embeddings, dim=0)

                # create chunk filename
                chunk_path = f"{base_path}_{start_idx}_{end_idx}_part{chunk_idx}{train_path.suffix}"
                torch.save(chunk_tensor, chunk_path)

                print(f"Saved chunk {chunk_idx} with {chunk_tensor.shape[0]} records to {chunk_path}")
                print(f"Time elapsed: {time.time() - time_start:.2f} seconds, Data processed: {i + 1}")

                train_embeddings = []
                torch.cuda.empty_cache()
                chunk_idx += 1
                time_start = time.time()

        # Save any remaining embeddings
        if train_embeddings:
            final_tensor = torch.cat(train_embeddings, dim=0)
            chunk_path = f"{base_path}_part{chunk_idx}{train_path.suffix}"
            torch.save(final_tensor, chunk_path)
            print(f"Saved final chunk {chunk_idx} with {final_tensor.shape[0]} records to {chunk_path}")
        print("End of extraction.")
    else:
        print(train_path, " exists, skipping operations.")

    if not os.path.exists(desc_path):
        desc_embeddings = []
        for i in range(len(desc_df)):
            text = desc_df['task_description'].values[i]
            encoded_input = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                padding=False,
                return_tensors='pt'
            ).to(device)
            # encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
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
    else:
        print(desc_path, " exists, skipping operations.")
    torch.cuda.empty_cache()


    if not os.path.exists(test_path):
        test_embeddings = []
        for i in range(len(test_df)):
            text = test_df['code'].values[i]
            encoded_input = tokenizer(
                text,  # Only one input at a time
                max_length=max_len,
                truncation=True,
                padding=False,
                return_tensors='pt'
            ).to(device)
            # encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
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
    else:
        print(test_path, " exists, skipping operations.")
    torch.cuda.empty_cache()


    if vis_df is not None:
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
    



def main():
    parser = argparse.ArgumentParser(description="Generate embeddings with different models")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Model name to use (e.g., gpt, code7b, code13b, llama7b, etc.)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--train", 
        type=str, 
        required=True, 
        help="Path to training dataframe (CSV/Parquet)"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        required=True, 
        help="Path to testing dataframe (CSV/Parquet)"
    )
    parser.add_argument(
        "--desc", 
        type=str, 
        required=True, 
        help="Path to description dataframe (CSV/Parquet)"
    )
    parser.add_argument(
        "--vis", 
        type=str, 
        default=None, 
        help="Optional path to visualization dataframe (CSV/Parquet)"
    )
    args = parser.parse_args()

    # Load dataframes
    def load_df(path):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        elif path.endswith(".parquet"):
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
    train_df = load_df(args.train)
    test_df = load_df(args.test)
    desc_df = load_df(args.desc)
    vis_df = load_df(args.vis) if args.vis else None

    device = torch.device(args.device)

    # Call your function
    combined_embedding(train_df, test_df, desc_df, vis_df, device, args.model)


if __name__ == "__main__":
    main()




