import torch
import time

def predict(test, model_lang, model_task, lang_y, task_y):
    num_test = len(test)
    step = 10
    pred_y_lang = torch.zeros(1)
    pred_y_task = torch.zeros(1)
    time_start = time.time()
    for i in range(0, num_test, step):
        predicted_1 = []
        predicted_2 = []
        if i + step >= num_test:
            predicted_1 = model_lang(test[i: num_test])
            predicted_2 = model_task(test[i: num_test])
        else:
            predicted_1 = model_lang(test[i: i+step])
            predicted_2 = model_task(test[i: i+step])
        if i == 0:
            pred_y_lang = predicted_1
            pred_y_task = predicted_2
        else:
            pred_y_lang = torch.cat((pred_y_lang, predicted_1), 0)
            pred_y_task = torch.cat((pred_y_task, predicted_2), 0)
        if i%1000==0:
            print('Time elapsed: {:.2f} seconds, Data predicted:{}'.format(time.time()-time_start, i))
            time_start = time.time()
    caculate_acc(pred_y_lang, lang_y, 'Programming Language')
    caculate_acc(pred_y_task, task_y, 'Programming Task')


def caculate_acc(pred_y, test_y, task_name):
    total_correct = (pred_y == test_y).sum().item()
    # Calculate the accuracy
    acc = 100 * total_correct / test_y.size(0)
    print("Accuracy of {} prediction: {:.2f}%".format(task_name, acc))


def predict_one(test_x, model, test_y, le):
    num_test = len(test_x)
    step = 10
    pred_y = torch.zeros(1)
    time_start = time.time()
    for i in range(0, num_test, step):
        predicted = []
        if i + step >= num_test:
            predicted = model(test_x[i: num_test])
        else:
            predicted = model(test_x[i: i+step])
        if i == 0:
            pred_y = predicted
        else:
            pred_y = torch.cat((pred_y, predicted), 0)
        if i%5000==0:
            print('Time elapsed: {:.2f} seconds, Data predicted:{}'.format(time.time()-time_start, i))
            time_start = time.time()
    caculate_acc(pred_y, test_y, 'combined labels')
    statstic(pred_y, test_y, le)
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