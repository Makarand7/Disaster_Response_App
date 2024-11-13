# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # merge datasets
    df = messages.merge(categories)
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[[1]]
    # extract column names
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:].astype(int)
    
    # drop the original categories column and concatenate with categories
    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument. \n\nExample: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db')

if __name__ == '__main__':
    main()
