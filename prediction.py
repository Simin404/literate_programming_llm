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
        if i + step > num_test:
            predicted_1 = model_lang(test[i, num_test])
            predicted_2 = model_task(test[i, num_test])
        else:
            predicted_1 = model_lang(test[i: i+step])
            predicted_2 = model_task(test[i: i+step])
        if i == 0:
            pred_y_lang = predicted_1
            pred_y_task = predicted_2
        else:
            pred_y_lang = torch.cat((pred_y_lang, predicted_1), 0)
            pred_y_task = torch.cat((pred_y_task, predicted_2), 0)
        if (i+1)%1000==0:
            print('Time elapsed: {:.2f} seconds, Data predicted:{}'.format(time.time()-time_start, i+1))
            time_start = time.time()
    caculate_acc(pred_y_lang, pred_y_task, lang_y, task_y)


def caculate_acc(pred_y_lang, pred_y_task, lang_y, task_y):
    total_correct_lang = (pred_y_lang == lang_y).sum().item()
    # Calculate the accuracy for this epoch
    acc_lang = 100 * total_correct_lang / lang_y.size(0)
    print('Accuracy of Language prediction:  {:.2f}%'.format(acc_lang))


    total_correct_task = (pred_y_task == task_y).sum().item()
    # Calculate the accuracy for this epoch
    acc_task = 100 * total_correct_task / task_y.size(0)
    print('Accuracy of Language prediction:  {:.2f}%'.format(acc_task))