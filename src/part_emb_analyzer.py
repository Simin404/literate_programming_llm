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

class EmbeddingAnalyzer:
    def __init__(self, model_name: str, classifier: str, indices_train: list, indices_test: list):
        self.model_name = model_name
        self.classifier = classifier
        self.indices_train = indices_train
        self.indices_test = indices_test       
        train_path = f'out/train/{self.model_name}_24892.pt'
        test_path = f'out/test/{self.model_name}_13786.pt'

        train_emb = torch.load(train_path, map_location="cpu")
        test_emb = torch.load(test_path, map_location="cpu")

        self.train_emb = torch.index_select(train_emb, 0, torch.tensor(indices_train))
        self.test_emb = torch.index_select(test_emb, 0, torch.tensor(indices_test))

    def preprocess_desc(self, train_df, test_df):
        if self.model_name == 'embedding_ada':
            self.train_emb, removed_train = self.remove_nan_rows(self.train_emb)
            train_df = train_df.drop(index=removed_train).reset_index(drop=True)

            self.test_emb, removed_test = self.remove_nan_rows(self.test_emb)
            test_df = test_df.drop(index=removed_test).reset_index(drop=True)

            logging.info('NaN values removed from embeddings.')
        return train_df, test_df
    
    # def predict_cnn(self, train_y_lang, test_y_lang, train_y_task, test_y_task):
    #     device = getting_device()
    #     # model_lang, acc_lang, _ = train_cnn_model(self.train_emb, train_y_lang, self.test_emb, test_y_lang, device)
    #     # model_task, acc_task, _ = train_cnn_model(self.train_emb, train_y_task, self.test_emb, test_y_task, device)

    #     logging.info(f'Accuracy -> Language: {acc_lang}')
    #     logging.info(f'Accuracy -> Task: {acc_task}')
    #     # return acc_lang, acc_task, acc_desc

    def analyze(self, train_df, test_df):
        logging.info(f'Model: {self.model_name}; Classifier: {self.classifier}')
        train_df, test_df = self.preprocess_desc(train_df, test_df)
        
        # Use the map_label method to encode labels
        train_y_lang, test_y_lang, _, _ = self.map_label(train_df['language'], test_df['language'])
        train_y_task, test_y_task, _, _ = self.map_label(train_df['task'], test_df['task'])

        train_y_task = torch.index_select(train_y_task, 0, torch.tensor(self.indices_train))
        train_y_lang = torch.index_select(train_y_lang, 0, torch.tensor(self.indices_train))

        test_y_task = torch.index_select(test_y_task, 0, torch.tensor(self.indices_test))
        test_y_lang = torch.index_select(test_y_lang, 0, torch.tensor(self.indices_test))

        model = self.select_classifier()

        if model is None:
            logging.error("Invalid classifier name!")
            return 0, 0

        return self.predict_and_evaluate(model, train_y_lang, test_y_lang, train_y_task, test_y_task)

    def select_classifier(self):
        if self.classifier == 'KNN':
            return KNeighborsClassifier(n_neighbors=10)  
        elif self.classifier == 'SVM':
            return SVC(C=100, degree=1, kernel='poly') 
        return None

    def predict_and_evaluate(self, model, train_y_lang, test_y_lang, train_y_task, test_y_task):
        acc_lang, _ = self.train_and_evaluate(model, self.train_emb, train_y_lang, self.test_emb, test_y_lang, "Language")
        acc_task, _ = self.train_and_evaluate(model, self.train_emb, train_y_task, self.test_emb, test_y_task, "Task")

        logging.info(f'Final Accuracy -> Language: {acc_lang:.4f}, Task: {acc_task:.4f}')
        return acc_lang, acc_task

    def train_and_evaluate(self, model, train_x, train_y, test_x, test_y, label: str):
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        return accuracy_score(test_y, y_pred), model


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
        # Find rows with any NaN values
        nan_mask = torch.isnan(tensor).any(dim=1)
        
        # Indices of rows to remove
        removed_indices = torch.where(nan_mask)[0].tolist()
        
        # Tensor without rows containing NaN
        cleaned_tensor = tensor[~nan_mask]
        
        return cleaned_tensor, removed_indices
