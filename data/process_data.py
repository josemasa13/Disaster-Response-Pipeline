import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Summary line
    
    Loads messages and categories datasets to merge them
    
    Parameters:
    messages_filepath(string): Relative path to the messages dataset
    categories_filepath(string) Relative path to the categories dataset
    
    Returns:
    df(DataFrame): Pandas dataframe containing datasets merged
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="left", on="id")
    return df


def clean_data(df):
    """
    Summary line
    
    Cleans the original dataframe based on specific rules
    
    Parameters:
    df(DataFrame): Pandas dataframe returned by the load_data function
    
    Returns:
    df(DataFrame): Pandas dataframe with it's data ready to feed a machine learning algorithm
    """
    # Split categories into separate category columns
    categories = df["categories"].str.split(";", expand=True)

    # Storing column names in a list
    first_row = categories.iloc[0]
    column_names = []
    for item in first_row:
        column_names.append(item.split("-")[0])
        
    # Assigning new column names to categories df
    categories.columns = column_names

    # Leaving just 1's and 0's for every record
    for column in categories.columns:
        categories[column] = categories[column].str.split("-").str[1]
        categories[column] = pd.to_numeric(categories[column])

    # Concatenating categories dataframe with the original dataframe
    df.drop(columns=["categories"], inplace = True)
    df = pd.concat([df, categories], axis=1)

    # Dropping duplicated records in the dataframe
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Summary line
    
    Stores the dataframe passed as parameter into the current directory as a database file
    
    Parameters:
    df(DataFrame): Pandas dataframe to be stored as a file
    database_filename(string): Name for the database file
    
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('Message', engine, index=False)
    pass  


def main():
    """
    Summary line
    
    Orchestrates the calls to previous functions to complete the ETL pipeline
    from extracting the data until loading it in a database file
    
    Parameters:
    None
    
    Returns: 
    None
    
    """
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