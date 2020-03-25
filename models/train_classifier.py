import sys
# import libraries
import pandas as pd
import nltk
import numpy as np
from sqlalchemy import create_engine
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')

def load_data(database_filepath):
    """
    Summary line
    
    Loads data from a database file to pandas DataFrame and splits it into
    feature and target variables
    
    Parameters:
    database_filepath(string): path in which the database file is located
    
    Returns:
    X(numpy array): Numpy array with the messages
    
    Y(numpy array): Numpy array of arrays with target values
    
    target_names(Index): Pandas Index with the target variables column names
    
    """
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("Message", engine)
    X = df['message'].values
    Y = df.drop(columns=["id", "message", "original", "genre"]).values
    target_names = df.drop(columns=["id", "message", "original", "genre"]).columns
    return X, Y, column_names


def tokenize(text):
    """
    Summary line
    
    tokenize function to split sentences into words, remove stop words
    and lemmatize them as part of the NLP pipeline
    
    Parameters:
    text(string): String to be tokenized
    
    Returns:
    clean_tokens(list): List of tokens derived from the text parameter

    """
    # Cleaning the text from punctuation and converting it all to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    
    # Removing stop words from the text
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    # Lemmatizer object
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatizing each token
    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token.strip(), pos="v")
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Summary line:
    
    Builds a machine learning model using a pipeline
    
    Parameters:
    None
    
    Returns:
    pipeline(Pipeline): Scikit learn's pipeline with the steps to fit a machine
    learning model
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Summary line:
    
    Predicts a new set of inputs with a model and evaluates
    these predictions using Scikit learn's classification
    report
    
    Parameters:
    model(Pipeline): Scikit learn's trained pipeline model
    
    X_test(numpy array): Numpy array with the test subset of the 
    X(feature values) numpy array returned by the load_data function
    
    Y_test(numpy array): Numpy array with the test subset of the 
    Y(target values) numpy array returned by the load_data function
    
    category_names(Index): Pandas index with the column names for the
    target variables
    
    Returns:
    None
    """
    Y_pred = model.predict(X_test)
    for i in range(0,len(category_names)):
        curr_test = [y[i] for y in Y_test]
        curr_pred = [y[i] for y in Y_pred]

        print(category_names[i])
        print(classification_report(curr_test, curr_pred))
        print()


def save_model(model, model_filepath):
    """
    Summary line:
    
    Saves a trained machine learning model into a pickle file
    
    Parameters:
    model(Pipeline): Scikit learn's pipeline trained model
    
    model_filepath(string): Path in which the pickle file will be stored
    """
    pickle.dump(model, open(model_filepath, "wb"))
    


def main():
    """
    Summary line
    
    Orchestrates the calls to previous functions to execute the machine learning pipeline
    
    Parameters:
    None
    
    Returns: 
    None
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()