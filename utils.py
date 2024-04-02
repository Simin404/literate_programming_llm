
from datasets import load_dataset
from collections import Counter
import os
import torch
import pandas as pd
import os
from os.path import isfile, join
from os import listdir


def file_to_df(file_path):
    tasks= [f for f in os.listdir(file_path) if not f.startswith('.') ]
    task_names=[]
    language_names=[]
    code_lists=[]
    for oneTask in tasks:
        oneTaskDir=file_path+"/"+oneTask
        languages=[f for f in os.listdir(oneTaskDir) if not f.startswith('.') ]
        for lang in languages:
            codeDir=test=oneTaskDir+"/"+lang
            if os.path.isdir(codeDir):
                onlyfiles = [f for f in listdir(codeDir) if isfile(join(codeDir, f))]
                for file_name in onlyfiles:
                    code_file_dir=codeDir+"/"+file_name
                    with open(code_file_dir, 'r') as file:
                        data = file.read()
                        task_names.append(oneTask)
                        language_names.append(lang)
                        code_lists.append(data)

    d = {'task': task_names, 'language': language_names,"code":code_lists}
    df = pd.DataFrame(data=d)
    df.to_csv("data/RosettaCodeData.csv")

def data_from_csv(file_path):
    all_df=pd.read_csv(file_path)
    return all_df

def split_data(df):
    df1 = df[df.groupby(['task','language']).transform('size') > 1]
    df_test = df1.groupby(['task','language']).sample(n = 1, random_state = 1)

    test_path = 'data/test_'+ str(df_test.shape[0])+'.csv'

    df_test.to_csv(test_path, index=False) 
    df_train = df1.drop(df_test.index)
    train_path = 'data/train_'+ str(df_train.shape[0])+'.csv'

    df_train.to_csv(train_path, index=False)
    return train_path, test_path


def load_data():
    # Load the Cakiki/Rosetta Code dataset
    all_data = load_dataset("cakiki/rosetta-code")['train']
    return all_data


def filter_data(all_data, languages = None, tasks = None, model='gpt'):
    part_data = all_data.filter(lambda example: example['language_name']in languages)
    part_data = part_data.filter(lambda example: example['task_name']in tasks)
    ## define output file name
    output_path = 'out/'+ model + '_'+str(len(part_data))+'.pt'
    print(output_path)
    return part_data, output_path


def getting_topN(all_data, top_N):
    
    # Getting topN languages that have the most records
    top_lang = Counter(all_data['language_name']).most_common(top_N)
    languages = [a for (a, b) in top_lang]
    print('Total languages included:', str(len(languages)), ', Example of top 5:', languages[0:5])

    # Getting topN tasks that have the most records
    top_tasks = Counter(all_data['task_name']).most_common(top_N)
    tasks = [a for (a, b) in top_tasks]
    print('Total tasks included:', str(len(tasks)), ', Example of top 5:', tasks[0:5])

    return languages, tasks


def getting_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)
    return device


def loading_embeddings(file_path):
    try:
        saved_emb = torch.load(file_path)
        saved_emb = saved_emb.cpu()
        print('Loading embeddings from:', file_path, ', Number of records:' +str(saved_emb.shape))
        return saved_emb
    except:
        print('File not found!')



