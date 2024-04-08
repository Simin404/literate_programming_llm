import os
import logging
import utils
import time
import pandas as pd
from prediction_w_log import analyze_data

def main():
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = f"../log/analysis_{timestamp}.log"
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Define path to search for dictionaries
    path = "../model/co-roberta"
    
    # Call the recursive function to get all models names
    models_names = get_folder_names(path)
    print(models_names)

    device = utils.getting_device()
    logger.info(f'Device: {device}')
    train_df =pd.read_csv('../data/train_24892.csv')
    test_df = pd.read_csv('../data/test_13786.csv')
    print(train_df.shape, test_df.shape)
    
    # Pass logger and dictionary names to predict function
    for model in models_names:
        if os.path.isfile('../out/'+model+'_24892.pt') and os.path.isfile('../out/'+model+'_13786.pt'): 
            print('Embedding exists, Load from file')
            train_file = '../out/'+model+'_24892.pt'
            test_file = '../out/'+model+'_13786.pt'
            analyze_data(model, train_df, test_df, device, logger, train_path = train_file, test_path = test_file, mode = 'two_label')
            # analyze_data(model, train_df, test_df, device, logger, train_path = train_file, test_path = test_file, mode = 'one_label')
        else:
            analyze_data(model, train_df, test_df, device, logger,  mode = 'emb')
            # train_file = '../out/'+model+'_24892.pt'
            # test_file = '../out/'+model+'_13786.pt'
            # analyze_data(model, train_df, test_df, device, logger, train_path = train_file, test_path = test_file, mode = 'one_label')



def get_folder_names(folder_path):
    folder_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    main()
