import torch
import time
import utils
import numpy as np
import pandas as pd
import extracting_embedding
from clustering import KNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def map_label(train_label, test_label):

    le = LabelEncoder()
    le=le.fit(test_label)

    train_label_encoding=le.transform(train_label)
    test_label_encoding=le.transform(test_label)

    train_y = torch.tensor(train_label_encoding)
    test_y = torch.tensor(test_label_encoding)

    # print(f"Original train count: {len(train_label)}, "
    #     f"test count: {len(test_label)}.")
    # print(f"After mapping: train count: {len(train_y)}, "
    #     f"test count: {len(test_y)}.")

    return train_y, test_y, le

def analyze_data(model_name, train_df, test_df, device, train_path = None, test_path = None, mode = 'two_label'):
    if mode == 'emb':
        train_path = extracting_embedding.extract_embedding(train_df, device, model=model_name, max_len = 100)
        test_path = extracting_embedding.extract_embedding(test_df, device, model=model_name, max_len = 100)
    
    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")

    if mode == 'one_label':
        train_df["combine"]=train_df["task"]+"@"+train_df["language"]
        test_df["combine"]=test_df["task"]+"@"+test_df["language"]
        train_y, test_y, le = map_label(train_df['combine'], test_df['combine'])

        knn_model = KNN(train_emb, train_y)
        
        pred_y = predict_one(test_emb, knn_model) 
        caculate_acc(pred_y, test_y, 'combined labels')
        statstic(pred_y, test_y, le)
        
    else:
        train_y_lang, test_y_lang, le_lang = map_label(train_df['language'], test_df['language'])
        train_y_task, test_y_task, le_task = map_label(train_df['task'], test_df['task'])
        
        knn_lang = KNN(train_emb, train_y_lang)
        knn_task = KNN(train_emb, train_y_task)
        
        pred_y_lang, pred_y_task = predict_two(test_emb, knn_lang, knn_task)
        dic_lang = acc_per(pred_y_lang, test_y_lang, le_lang,)
        dic_task = acc_per(pred_y_task, test_y_task, le_task)

        caculate_acc(pred_y_lang, test_y_lang, 'Programming Language')
        caculate_acc(pred_y_task, test_y_task, 'Programming Task') 
        return dic_lang, dic_task
    
def part_data_analysis(languages, train_df, test_df, train_file, test_file):
    indices_train = train_df.index[train_df['language'].isin(languages)].tolist()
    indices_test = test_df.index[test_df['language'].isin(languages)].tolist()

    train_y_lang, test_y_lang, le_lang = map_label(train_df['language'], test_df['language'])
    train_y_task, test_y_task, le_task = map_label(train_df['task'], test_df['task'])

    train_emb =  torch.load(train_file, map_location="cpu")
    test_emb = torch.load(test_file, map_location="cpu")

    train_emb_sub = torch.index_select(train_emb, 0, torch.tensor(indices_train))
    train_y_task_sub = torch.index_select(train_y_task, 0, torch.tensor(indices_train))
    train_y_lang_sub = torch.index_select(train_y_lang, 0, torch.tensor(indices_train))

    test_emb_sub = torch.index_select(test_emb, 0, torch.tensor(indices_test))
    test_y_task_sub = torch.index_select(test_y_task, 0, torch.tensor(indices_test))
    test_y_lang_sub = torch.index_select(test_y_lang, 0, torch.tensor(indices_test))

    knn_codebert_lang_sub = KNN(train_emb_sub, train_y_lang_sub)
    knn_codebert_task_sub = KNN(train_emb_sub, train_y_task_sub)

    pred_y_lang, pred_y_task = predict_two(test_emb_sub, knn_codebert_lang_sub, knn_codebert_task_sub)
    caculate_acc(pred_y_lang, test_y_lang_sub, 'Programming Language')
    caculate_acc(pred_y_task, test_y_task_sub, 'Programming Task')


def predict_two(test, model_lang, model_task):
    num_test = len(test)
    step = 10
    time_start = time.time()

    pred_y_lang = []
    pred_y_task = []

    for i in range(0, num_test, step):
        # Calculate the end index of the current slice
        end_index = min(i + step, num_test)

        predicted_1 = model_lang(test[i:end_index])
        predicted_2 = model_task(test[i:end_index])

        if i == 0:
            pred_y_lang = predicted_1
            pred_y_task = predicted_2
        else:
            pred_y_lang = torch.cat((pred_y_lang, predicted_1), 0)
            pred_y_task = torch.cat((pred_y_task, predicted_2), 0)

        if i % 5000 == 0:
            print('Time elapsed: {:.2f} seconds, Data predicted: {}'.format(time.time() - time_start, i))
            time_start = time.time()
    return pred_y_lang, pred_y_task




def caculate_acc(pred_y, test_y, task_name):
    total_correct = (pred_y == test_y).sum().item()
    # Calculate the accuracy
    acc = 100 * total_correct / test_y.size(0)
    print("Accuracy of {} prediction: {:.2f}%".format(task_name, acc))


def acc_per(pred_y, real_y, le):
    languages = list(torch.unique(real_y))
    lang_dic = {}
    for number in languages:
        indice = ((real_y==number).nonzero().squeeze())
        r_y = torch.index_select(real_y, 0, indice)
        p_y = torch.index_select(pred_y, 0, indice)
        acc = (p_y == r_y).sum().item() / r_y.size(0)
        a = number.numpy()
        language=le.inverse_transform([a])
        lang_dic[language[0]] = acc
    return lang_dic

def count_data(data):
    bins = {'=0.0': [], '(0.0 - 0.1]': [], '(0.1 - 0.2]': [], '(0.2 - 0.3]': [], '(0.3 - 0.4]': [],
            '(0.4 - 0.5]': [], '(0.5 - 0.6]': [], '(0.6 - 0.7]': [], '(0.7 - 0.8]': [], '(0.8 - 0.9]': [], '(0.9 - 1.0]': []}
    
    for key, value in data.items():
        if value == 0.0:
            bins['=0.0'].append(key)
        elif 0.0 < value <= 0.1:
            bins['(0.0 - 0.1]'].append(key)
        elif 0.1 < value <= 0.2:
            bins['(0.1 - 0.2]'].append(key)
        elif 0.2 < value <= 0.3:
            bins['(0.2 - 0.3]'].append(key)
        elif 0.3 < value <= 0.4:
            bins['(0.3 - 0.4]'].append(key)
        elif 0.4 < value <= 0.5:
            bins['(0.4 - 0.5]'].append(key)
        elif 0.5 < value <= 0.6:
            bins['(0.5 - 0.6]'].append(key)
        elif 0.6 < value <= 0.7:
            bins['(0.6 - 0.7]'].append(key)
        elif 0.7 < value <= 0.8:
            bins['(0.7 - 0.8]'].append(key)
        elif 0.8 < value <= 0.9:
            bins['(0.8 - 0.9]'].append(key)
        elif 0.9 < value <= 1.0:
            bins['(0.9 - 1.0]'].append(key)   
    return bins

def plot_count(bins, task_name):
    labels = bins.keys()
    counts = [len(i) for i in bins.values()]

    plt.bar(labels, counts, color=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow', 'black'])
    plt.xlabel('Value Ranges')
    plt.ylabel('Frequency')
    plt.title(task_name + ' Prediction Accuarcy Analysis')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('image/'+task_name+'.png')

def plot_multi_count(analysis, data_labels):
    labels = analysis[0].keys()
    width = 0.1
    fig, ax = plt.subplots()
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow', 'black']
    x = np.arange(len(labels))

    for i, (bins, label) in enumerate(zip(analysis, data_labels)):
        counts = [len(i) for i in bins.values()]
        ax.bar(x + i * width, counts, width=width, color=colors[i], label=label)

    ax.set_xlabel('Value Ranges')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Accuarcy Analysis')
    ax.set_xticks(x + width * (len(analysis) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.show()


def predict_one(test_x, model):
    num_test = len(test_x)
    step = 10
    pred_y = []
    time_start = time.time()
    for i in range(0, num_test, step):
        end_index = min(i + step, num_test)

        predicted = model(test_x[i:end_index])

        if i == 0:
            pred_y = predicted
        else:
            pred_y = torch.cat((pred_y, predicted), 0)
        if i%5000==0:
            print('Time elapsed: {:.2f} seconds, Data predicted:{}'.format(time.time()-time_start, i))
            time_start = time.time()
    return pred_y


def statstic(pred_y, test_y, le):
    right=0
    right_language=0
    right_task=0
    for i in range(len(test_y)):
        true_label=test_y[i]
        predict_label=pred_y[i]
        if true_label==predict_label:
            right+=1
        else:
            t_reverse=le.inverse_transform(true_label.numpy().ravel())[0].split("@")
            p_reverse=le.inverse_transform(predict_label.numpy().ravel())[0].split("@")
            if t_reverse[0]==p_reverse[0]:
                right_task+=1
            if t_reverse[1]==p_reverse[1]: 
                right_language+=1
    print('Total Number:', test_y.size(0))
    print('All right:', right)
    print('Language correct:', right_language)
    print('Task correct:', right_task)