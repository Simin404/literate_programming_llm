import torch
import logging
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from src.cnn import *
from src.utils import *

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log/{timestamp}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

class No_Para_Analyzer:
    def __init__(self, model_name: str, classifier: str, params):
        self.model_name = model_name
        self.classifier = classifier
        train_path = f'out/train/{self.model_name}_1725993.pt'
        test_path = f'out/test/{self.model_name}_1824.pt'
        desc_path = f'out/desc/{self.model_name}_55.pt'
        self.train_emb = torch.load(train_path, map_location="cpu")
        self.test_emb = torch.load(test_path, map_location="cpu")
        self.desc_emb = torch.load(desc_path, map_location="cpu")
        self.params = params

    def preprocess_desc(self, train_df, test_df, desc_df):
        if self.model_name == 'embedding_ada':
            self.train_emb, removed_train = self.remove_nan_rows(self.train_emb)
            train_df = train_df.drop(index=removed_train).reset_index(drop=True)

            self.test_emb, removed_test = self.remove_nan_rows(self.test_emb)
            test_df = test_df.drop(index=removed_test).reset_index(drop=True)

            self.desc_emb, removed_desc = self.remove_nan_rows(self.desc_emb)
            desc_df = desc_df.drop(index=removed_desc).reset_index(drop=True)
            logging.info('NaN values removed from embeddings.')
        return train_df, test_df, desc_df
    
    def predict_cnn(self, train_y_lang, test_y_lang, train_y_task, test_y_task, desc_y_task):
        device = getting_device()
        model_lang, acc_lang, _ = train_cnn_model(self.train_emb, train_y_lang, self.test_emb, test_y_lang, device, X_desc=None, y_desc=None, params=self.params)
        model_task, acc_task, acc_desc = train_cnn_model(self.train_emb, train_y_task, self.test_emb, test_y_task, device, X_desc = self.desc_emb, y_desc=desc_y_task, params=self.params)

        logging.info(f'Accuracy -> Language: {acc_lang}')
        logging.info(f'Accuracy -> Task: {acc_task}')
        logging.info(f'Accuracy -> Desc: {acc_desc}')

    def analyze(self, train_df, test_df, desc_df):
        params_str = self.params if self.params else 'default'
        logging.info(f'Model: {self.model_name}; Classifier: {self.classifier}; Parameters: {params_str}')

        train_df, test_df, desc_df = self.preprocess_desc(train_df, test_df, desc_df)
        logging.info(f'train_df: {train_df.shape}; test_df: {test_df.shape}; self.train_emb: {self.train_emb.shape}')
        train_y_lang, test_y_lang, _ , _ = self.map_label(train_df['language'], test_df['language'])
        train_y_task, test_y_task, desc_y_task, _ = self.map_label(train_df['task'], test_df['task'], desc_df['task'])

        if self.classifier == 'CNN':
            self.predict_cnn(train_y_lang, test_y_lang, train_y_task, test_y_task, desc_y_task)
            return 0, 0, 0

        model = self.get_classifier_with_params()
        if model is None:
            logging.error("Invalid classifier name or parameters!")
            return 0, 0, 0

        return self.predict_and_evaluate(model, train_y_lang, test_y_lang, train_y_task, test_y_task, desc_y_task)


    def get_classifier_with_params(self):
        if self.classifier == 'KNN':
            return KNeighborsClassifier(**self.params) if self.params else KNeighborsClassifier()
        elif self.classifier == 'SVM':
            return SVC(**self.params) if self.params else SVC()
        elif self.classifier == 'Forest':
            return RandomForestClassifier(**self.params) if self.params else RandomForestClassifier(random_state=42)
        return None


    def predict_and_evaluate(self, model, train_y_lang, test_y_lang, train_y_task, test_y_task, desc_y_task):
        acc_lang = self.train_and_evaluate(model, self.train_emb, train_y_lang, self.test_emb, test_y_lang, "Language")
        acc_task, model_task = self.train_and_evaluate_with_model(model, self.train_emb, train_y_task, self.test_emb, test_y_task, "Task")

        y_pred = model_task.predict(self.desc_emb)
        acc_desc = accuracy_score(desc_y_task, y_pred)

        logging.info(f'Final Accuracy -> Language: {acc_lang:.4f}, Task: {acc_task:.4f}, Desc: {acc_desc:.4f}')
        return acc_lang, acc_task, acc_desc


    def train_and_evaluate(self, model, train_x, train_y, test_x, test_y, label: str):
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        acc = accuracy_score(test_y, y_pred)
        logging.info(f"{label}: Accuracy -> {acc:.4f}")
        return acc

    def train_and_evaluate_with_model(self, model, train_x, train_y, test_x, test_y, label: str):
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        acc = accuracy_score(test_y, y_pred)
        logging.info(f"{label}: Accuracy -> {acc:.4f}")
        return acc, model

    @staticmethod
    def map_label(train_label, test_label, desc_label=None):
        le = LabelEncoder()
        le.fit(test_label)  # Fit the encoder on the test labels

        # Transform labels and convert them to PyTorch tensors
        train_label_encoding = le.transform(train_label)
        test_label_encoding = le.transform(test_label)

        train_y = torch.tensor(train_label_encoding, dtype=torch.long)
        test_y = torch.tensor(test_label_encoding, dtype=torch.long)

        # Encode description labels if provided
        if desc_label is not None:
            desc_label_encoding = le.transform(desc_label)
            desc_y = torch.tensor(desc_label_encoding, dtype=torch.long)
        else:
            desc_y = None

        return train_y, test_y, desc_y, le

    @staticmethod
    def remove_nan_rows(tensor):
        nan_mask = torch.isnan(tensor).any(dim=1)
        removed_indices = torch.where(nan_mask)[0].tolist()
        cleaned_tensor = tensor[~nan_mask]
        
        return cleaned_tensor, removed_indices
