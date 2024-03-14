import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import Dataset
import pandas as pd
import logging
import time
import utils
import os

# Configure logging to write to a file
import transformers
transformers.logging.set_verbosity_info()

log_file_path = 'log/training.log'
logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='w',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def preprocess_dataframe(df):
    """
    Preprocesses a DataFrame containing a column named 'code'.
    Splits the 'code' column by newline character, removes empty lines,
    and returns a new DataFrame with the preprocessed 'code' column.
    """
    df['code'] = df['code'].str.split('\n')
    df = df.explode("code")
    df['code'] = df['code'].replace('', np.nan)
    df = df.dropna()
    return df

def tokenize_dataset(df, tokenizer, device):
    """
    Tokenizes the 'code' column in the DataFrame using a Roberta tokenizer.
    Returns a new Dataset object with the tokenized 'code' column.
    """
    new_dataset = Dataset.from_pandas(df)
    
    tokenized_dataset = new_dataset.map(lambda x: tokenizer(x['code'], truncation=True, padding=True, max_length=512), 
            num_proc=16)
    return tokenized_dataset

def main():
    logger = logging.getLogger(__name__)

    # Set the device to GPU if available, otherwise use CPU
    device = utils.getting_device()
    logger.info(f"Using device: {device}")

    # Read CSV file and extract 'code' column
    df = pd.read_csv('data/rosettaCodeByLine.csv')
    logger.info(f"Original shape: {df.shape}")

    # Preprocess DataFrame
    #df = preprocess_dataframe(df)
    #logger.info(f"Processed shape: {df.shape}")
    #df.to_csv("data/rosettaCodeByLine.csv")    


    # Initialize the RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_length=512)
    model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)

    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(df, tokenizer, device)


    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    # Define training arguments
    training_args = TrainingArguments(
                    output_dir = "co-roberta",
                    overwrite_output_dir=True,
                    num_train_epochs=4,
                    per_device_train_batch_size=32,
                    save_steps=10000,
                    save_total_limit=2,
                    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    total_start_time = time.time()
    for epoch in range(training_args.num_train_epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")

        # Perform training
        trainer.train()

        # Calculate elapsed time for the epoch
        epoch_elapsed_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1} took: {epoch_elapsed_time:.2f} seconds")

    # Save the model
    model.save_pretrained("co-roberta")
    tokenizer.save_pretrained("co-roberta")

    # Calculate total training time
    total_elapsed_time = time.time() - total_start_time
    logger.info(f"Total training time: {total_elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
