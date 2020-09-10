import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    loads two dataframes from files, merges them together
    :param messages_filepath:
    :param categories_filepath:
    :return: merged dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    replace string column with 0-1 valued columns per category
    drop duplicates
    :param df:
    :return: cleaned df
    """
    #create columns from strings, splitting by ';'
    categories = df.categories.str.split(';', expand=True)

    #extract columns names from the first row
    row = categories.iloc[0, :]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    #replace values with 0 and 1 values from the strings
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    #drop old columns, append new columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    laod dataframe to database file
    :param df:
    :param database_filename:
    :return:
    """
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()