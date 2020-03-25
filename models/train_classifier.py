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
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table("Message", engine)
    X = df['message'].values
    Y = df.drop(columns=["id", "message", "original", "genre"]).values
    return X, Y, df.drop(columns=["id", "message", "original", "genre"]).columns


def tokenize(text):
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
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(0,len(category_names)):
        curr_test = [y[i] for y in Y_test]
        curr_pred = [y[i] for y in Y_pred]

        print(category_names[i])
        print(classification_report(curr_test, curr_pred))
        print()


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))
    


def main():
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