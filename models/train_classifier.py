import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
import nltk.tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def load_data(database_filepath):
    """
    read data from database, extract data and labels
    :param database_filepath:
    :return: data, labels, names of categories
    """
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).values
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    """
    remove characters but letters and numbers, normalize, tokenize, lemmatize, remove stopwords
    :param text:
    :return: list of tokens
    """
    words = nltk.tokenize.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return tokens


def build_model():
    """
    model pipeline. includes count vectorizer, tfidf and classification estimator
    :return: pipeline
    """
    pipeline = Pipeline([('count', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    run clasisfication report for each category
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    Y_pred = model.predict(X_test)
    for i in range(len(Y_pred[0])):
        print(category_names[i])
        print(classification_report(np.transpose(Y_test)[i], np.transpose(Y_pred)[i]))


def save_model(model, model_filepath):
    """
    save trained model to file
    :param model:
    :param model_filepath:
    :return:
    """
    p = open(model_filepath, "wb")
    pickle.dump(model, p)
    p.close()


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