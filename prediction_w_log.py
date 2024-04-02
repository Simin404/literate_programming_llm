import torch
import time
import utils
import pandas as pd
import extracting_embedding
from clustering import KNN
from sklearn.preprocessing import LabelEncoder
import logging


def map_label(train_label, test_label, logger):
    le = LabelEncoder()
    le = le.fit(test_label)

    train_label_encoding = le.transform(train_label)
    test_label_encoding = le.transform(test_label)

    train_y = torch.tensor(train_label_encoding)
    test_y = torch.tensor(test_label_encoding)

    logger.info(f"Original train count: {len(train_label)}, "
                f"test count: {len(test_label)}.")
    logger.info(f"After mapping: train count: {len(train_y)}, "
                f"test count: {len(test_y)}.")

    return train_y, test_y, le

def analyze_data(model_name, train_df, test_df, device, logger, train_path=None, test_path=None, mode='two_label'):

    logger.info('THE MODEL ANALYZED:', model_name)
    if mode == 'emb':
        train_path = extracting_embedding.extract_embedding(train_df, device, model=model_name, max_len=100)
        test_path = extracting_embedding.extract_embedding(test_df, device, model=model_name, max_len=100)

    train_emb = torch.load(train_path, map_location="cpu")
    test_emb = torch.load(test_path, map_location="cpu")

    if mode == 'one_label':
        train_df["combine"] = train_df["task"] + "@" + train_df["language"]
        test_df["combine"] = test_df["task"] + "@" + test_df["language"]
        train_y, test_y, le = map_label(train_df['combine'], test_df['combine'])

        knn_model = KNN(train_emb, train_y)

        predict_one(test_emb, knn_model, test_y, le, logger)

    else:
        train_y_lang, test_y_lang, _ = map_label(train_df['language'], test_df['language'])
        train_y_task, test_y_task, _ = map_label(train_df['task'], test_df['task'])

        knn_lang = KNN(train_emb, train_y_lang)
        knn_task = KNN(train_emb, train_y_task)

        predict_two(test_emb, knn_lang, knn_task, test_y_lang, test_y_task, logger)


def predict_two(test, model_lang, model_task, lang_y, task_y, logger):
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
            logger.info('Time elapsed: {:.2f} seconds, Data predicted: {}'.format(time.time() - time_start, i))
            time_start = time.time()

    caculate_acc(pred_y_lang, lang_y, 'Programming Language', logger)
    caculate_acc(pred_y_task, task_y, 'Programming Task', logger)


def caculate_acc(pred_y, test_y, task_name, logger):
    total_correct = (pred_y == test_y).sum().item()
    # Calculate the accuracy
    acc = 100 * total_correct / test_y.size(0)
    logger.info("Accuracy of {} prediction: {:.2f}%".format(task_name, acc))


def predict_one(test_x, model, test_y, le, logger):
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
        if i % 5000 == 0:
            logger.info('Time elapsed: {:.2f} seconds, Data predicted:{}'.format(time.time() - time_start, i))
            time_start = time.time()

    caculate_acc(pred_y, test_y, 'combined labels', logger)
    statstic(pred_y, test_y, le, logger)
    return pred_y


def statstic(pred_y, test_y, le, logger):
    right = 0
    right_language = 0
    right_task = 0
    for i in range(len(test_y)):
        true_label = test_y[i]
        predict_label = pred_y[i]
        if true_label == predict_label:
            right += 1
        else:
            t_reverse = le.inverse_transform(true_label.numpy().ravel())[0].split("@")
            p_reverse = le.inverse_transform(predict_label.numpy().ravel())[0].split("@")
            if t_reverse[0] == p_reverse[0]:
                right_task += 1
            if t_reverse[1] == p_reverse[1]:
                right_language += 1
    logger.info('Total Number: {}'.format(test_y.size(0)))
    logger.info('All right: {}'.format(right))
    logger.info('Language correct: {}'.format(right_language))
    logger.info('Task correct: {}'.format(right_task))
