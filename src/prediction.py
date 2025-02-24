import torch
import time
import numpy as np
import pandas as pd
from src.part_emb_analyzer import EmbeddingAnalyzer
import pickle
from src.extracting_embedding import extract_embedding
from src.clustering import KNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def map_label(train_label, test_label, desc_label=None):

    le = LabelEncoder()
    le=le.fit(test_label)

    train_label_encoding=le.transform(train_label)
    test_label_encoding=le.transform(test_label)

    train_y = torch.tensor(train_label_encoding)
    test_y = torch.tensor(test_label_encoding)
    desc_y = torch.tensor(0)

    if desc_label != None:
        desc_label_encoding=le.transform(desc_label)
        desc_y = torch.tensor(desc_label_encoding)
    
    return train_y, test_y, desc_y, le

    # print(f"Original train count: {len(train_label)}, "
    #     f"test count: {len(test_label)}.")
    # print(f"After mapping: train count: {len(train_y)}, "
    #     f"test count: {len(test_y)}.")


def append_pickle(path, app_dic):
    try:
        with open(path, "rb") as file:
            existing_data = pickle.load(file)
    except (FileNotFoundError, EOFError):
        existing_data = []
    existing_data.append(app_dic)
    # Save back the updated data
    with open(path, "wb") as file:
        pickle.dump(existing_data, file)
    print("Data appended successfully!")


def remove_nan_rows(tensor):
    # Find rows with any NaN values
    nan_mask = torch.isnan(tensor).any(dim=1)
    
    # Indices of rows to remove
    removed_indices = torch.where(nan_mask)[0].tolist()
    
    # Tensor without rows containing NaN
    cleaned_tensor = tensor[~nan_mask]
    
    return cleaned_tensor, removed_indices


def analyze(model_name, train_df, test_df, desc_df, classifier = 'knn'):
    train_path = 'out/'+model_name+'_24892.pt'
    test_path = 'out/'+model_name+'_13786.pt'
    desc_path = 'out/'+model_name+'_532.pt'
   
    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")
    desc_emb = torch.load(desc_path, map_location="cpu")

    train_y_lang, test_y_lang, _,  le_lang = map_label(train_df['language'], test_df['language'])
    train_y_task, test_y_task, desc_y_task, le_task = map_label(train_df['task'], test_df['task'], desc_df['task'])

    if classifier == 'knn':
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(2, 5)}

    elif classifier == 'SVM':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'degree': [2, 3, 4],
            'kernel': ['poly']
        }
        model = SVC()
    elif classifier == 'SVM':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'degree': [2, 3, 4],
            'kernel': ['poly']
        }
        model = SVC()
    elif classifier == 'forest':
        param_grid = {
            'n_estimators': [10, 50, 100],        # Number of trees
            'max_depth': [None, 5, 10],           # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],      # Minimum samples required to split
            'min_samples_leaf': [1, 2, 4]         # Minimum samples required per leaf
        }
        model = RandomForestClassifier(random_state=42)
    else:
        print('Wrong classifier name, please verify!')
    grid_search_lang = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search_lang.fit(train_emb, train_y_lang)   
    print("Classifier: {}; Language;  Best parameters: {}".format(classifier, grid_search_lang.best_params_))
    best_lang = grid_search_lang.best_params_['n_neighbors']
    y_pred_lang = best_lang.predict(test_emb)
    acc_lang = accuracy_score(test_y_lang, y_pred_lang)

    grid_search_task = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search_task.fit(train_emb, train_y_task)
    print("Classifier: {}; Task;  Best parameters: {}".format(classifier, grid_search_task.best_params_))
    best_task = grid_search_task.best_params_['n_neighbors']
    y_pred_task = best_task.predict(test_emb)
    acc_task = accuracy_score(test_y_task, y_pred_task)

    y_pred_desc = best_task.predict(desc_emb)
    acc_desc = accuracy_score(desc_y_task, y_pred_desc)
    return acc_lang, acc_task, acc_desc
    

def analyze_rf(model_name, train_df, test_df, device, train_path = None, test_path = None, mode = 'two_label'):
    if mode == 'emb':
        train_path = extract_embedding(train_df, device, model=model_name)
        test_path = extract_embedding(test_df, device, model=model_name)
    
    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")


    train_y_lang, test_y_lang, le_lang = map_label(train_df['language'], test_df['language'])
    train_y_task, test_y_task, le_task = map_label(train_df['task'], test_df['task'])
    
    param_grid = {
        'n_estimators': [10, 50, 100],        # Number of trees
        'max_depth': [None, 5, 10],           # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],      # Minimum samples required to split
        'min_samples_leaf': [1, 2, 4]         # Minimum samples required per leaf
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search_lang = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_lang.fit(train_emb, train_y_lang)
    best_rf_lang = grid_search_lang.best_estimator_
    print("Best Hyperparameters for lang:", grid_search_lang.best_params_)

    grid_search_task = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_task.fit(train_emb, train_y_task)
    best_rf_task = grid_search_task.best_estimator_
    print("Best Hyperparameters for task:", grid_search_task.best_params_)


    pred_y_lang = best_rf_lang.predict(test_emb)
    pred_y_task = best_rf_task.predict(test_emb)

    dic_lang = acc_per(pred_y_lang, test_y_lang, le_lang,)
    dic_task = acc_per(pred_y_task, test_y_task, le_task)

    acc_lang = accuracy_score(test_y_lang, pred_y_lang)
    acc_task = accuracy_score(test_y_task, pred_y_task)
    
    print("Classifier: Random Forest; Language testset accuracy:{:.2%}".format(acc_lang))
    print("Classifier: Random Forest; Task testset accuracy:{:.2%}".format(acc_task))

    return acc_lang, acc_task, dic_lang, dic_task


def analyze_svm(model_name, train_df, test_df, device, train_path = None, test_path = None, mode = 'two_label', kernal = 'rbf', c = 1.0):
    if mode == 'emb':
        train_path = extract_embedding(train_df, device, model=model_name)
        test_path = extract_embedding(test_df, device, model=model_name)
    
    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")

    if mode == 'one_label':
        train_df["combine"]=train_df["task"]+"@"+train_df["language"]
        test_df["combine"]=test_df["task"]+"@"+test_df["language"]
        train_y, test_y, le = map_label(train_df['combine'], test_df['combine'])

        svm_model = SVC(kernel=kernal, C=c)
        svm_model.fit(train_emb, train_y)
        
        pred_y = svm_model.predict(test_emb)
        caculate_acc(pred_y, test_y, 'combined labels')
        statstic(pred_y, test_y, le)
        
    else:
        train_y_lang, test_y_lang, le_lang = map_label(train_df['language'], test_df['language'])
        train_y_task, test_y_task, le_task = map_label(train_df['task'], test_df['task'])
        
        svm_model_lang = SVC(kernel=kernal, C=c)
        svm_model_task = SVC(kernel=kernal, C=c)

        # Fit the model on the training data
        svm_model_lang.fit(train_emb, train_y_lang)
        svm_model_task.fit(train_emb, train_y_task)
        
        pred_y_lang = svm_model_lang.predict(test_emb)
        pred_y_task = svm_model_task.predict(test_emb)

        # dic_lang = acc_per(pred_y_lang, test_y_lang, le_lang,)
        # dic_task = acc_per(pred_y_task, test_y_task, le_task)

        caculate_acc(pred_y_lang, test_y_lang, 'Programming Language')
        caculate_acc(pred_y_task, test_y_task, 'Programming Task') 
        # return dic_lang, dic_task


def analyze_mlp(model_name, train_df, test_df, device, train_path = None, test_path = None, mode = 'two_label'):
    if mode == 'emb':
        train_path = extract_embedding(train_df, device, model=model_name)
        test_path = extract_embedding(test_df, device, model=model_name)
    
    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")

    if mode == 'one_label':
        train_df["combine"]=train_df["task"]+"@"+train_df["language"]
        test_df["combine"]=test_df["task"]+"@"+test_df["language"]
        train_y, test_y, le = map_label(train_df['combine'], test_df['combine'])

        mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        mlp_model.fit(train_emb, train_y)
        
        pred_y = predict_one(test_emb, mlp_model) 
        caculate_acc(pred_y, test_y, 'combined labels')
        statstic(pred_y, test_y, le)
        
    else:
        train_y_lang, test_y_lang, le_lang = map_label(train_df['language'], test_df['language'])
        train_y_task, test_y_task, le_task = map_label(train_df['task'], test_df['task'])
        
        mlp_model_lang = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        mlp_model_task = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        # Fit the model on the training data
        mlp_model_lang.fit(train_emb, train_y_lang)
        mlp_model_task.fit(train_emb, train_y_task)
        
        pred_y_lang = mlp_model_lang.predict(test_emb)
        pred_y_task = mlp_model_task.predict(test_emb)

        # dic_lang = acc_per(pred_y_lang, test_y_lang, le_lang,)
        # dic_task = acc_per(pred_y_task, test_y_task, le_task)

        caculate_acc(pred_y_lang, test_y_lang, 'Programming Language')
        caculate_acc(pred_y_task, test_y_task, 'Programming Task') 
        # return dic_lang, dic_task

def analyze_data(model_name, train_df, test_df, device, train_path = None, test_path = None, mode = 'two_label'):
    if mode == 'emb':
        train_path = extract_embedding(train_df, device, model=model_name)
        test_path = extract_embedding(test_df, device, model=model_name)
    
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
    

def analyze_descriprion(train_df, test_df, train_path = None, test_path = None, classifier = 'KNN', k = None, cv = None):

    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")

    train_emb, removed_train = remove_nan_rows(train_emb)
    test_emb, removed_test = remove_nan_rows(test_emb)

    train_df = train_df.drop(index=removed_train)
    train_df.reset_index(drop=True, inplace=True)

    test_df = test_df.drop(index=removed_test)
    test_df.reset_index(drop=True, inplace=True)

    train_y_task, test_y_task, le_task = map_label(train_df['task'], test_df['task'])
    if classifier == 'KNN':
        my_classifier = KNN(train_emb, train_y_task)
        pred_y_task = predict_one(test_emb, my_classifier)
    elif classifier == 'SVM':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'degree': [2, 3, 4],
            'kernel': ['poly']
        }
        grid_search = GridSearchCV(SVC(), param_grid, cv=cv)
        grid_search.fit(train_emb, train_y_task)
        print("Best parameters:", grid_search.best_params_)
        # my_classifier = SVC(kernel=k, C=c)
        # my_classifier.fit(train_emb, train_y_task)
        my_classifier = grid_search.best_estimator_
        pred_y_task = my_classifier.predict(test_emb)
    elif classifier == 'forest':
        pass
    dic_task = acc_per(pred_y_task, test_y_task, le_task)
    accuracy = accuracy_score(test_y_task, pred_y_task)
    print("Classifier: {}: Testset accuracy:{:.2%}".format(classifier, accuracy))
    return accuracy, dic_task
    
def part_data_analysis(tasks, languages, train_df, test_df):

    indices_train = train_df.index[train_df['task'].isin(tasks) & train_df['language'].isin(languages)].tolist()
    indices_test = test_df.index[test_df['task'].isin(tasks) & test_df['language'].isin(languages)].tolist()

    model_names = ['bert', 'gpt', 'roberta', 'falcon7b', 'falcon11b',  'falcon40b', 'llama7b', 'llama8b', 'llama13b', 'llama70b','embedding_ada', 'embedding_small', 'embedding_large', 'codebert', 'codegpt', 'code7b', 'code13b', 'code34b', 'code70b']
    classifier_names = ['KNN', 'SVM']
    goals = ['lang', 'task']
    r_dic = {model: {classifier: {goal: None for goal in goals} for classifier in classifier_names} for model in model_names}

    for c in classifier_names:
        for m in model_names:
            if r_dic[m][c][goals[0]] is None:
                analyzer = EmbeddingAnalyzer(model_name = m , classifier = c, indices_train = indices_train, indices_test = indices_test)
                acc1, acc2 = analyzer.analyze(train_df, test_df)
                r_dic[m][c][goals[0]], r_dic[m][c][goals[1]] = round(acc1, 4), round(acc2, 4)
                print('Model: {}; Classifier: {}'.format(m, c))
                print('--------------------------------------------------------')
            else:
                print('Pass as already caculated!')
    out_path = "out/experiment2/"+str(len(tasks))+".pkl"
    append_pickle(out_path, r_dic)
    return r_dic


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
            print('Time elapsed: {:.2f} seconds, Data predicted: {}'.format(time.time() - time_start, i+1))
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

def plot_multi_count(analysis, data_labels, task_name):
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
    ax.set_xticks(x + width * (len(analysis) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.savefig('image/'+task_name+'.png')


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

    print('Time elapsed: {:.2f} seconds, Data predicted:{}'.format(time.time()-time_start, i+1))
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


