# Disaster Response Pipeline Project

### Project motivation

Main goal of the project is classifying messages by categories. Given a tweet or text message from real life disaster, built model assigns several of pre-defined categories (36 in total) to it.

Source data includes message raw text and pre-assigned categories for each of the message. Data is then used to train multi output classifier. 

All raw data is first processed, normalized and cleaned. Count vectorizer and TFIDF are used to prepare features for the model to train on.

Classifier being used is Random Forest, which showed itself as the best in this particular scenario.
Average f1 score is 0.92.

Working model is presented via a web app where user can input message text which will further be classified.

### Project description:

- Data folder 
Source csv files disaster_messages and disaster_categories.
Script for ETL pipeline (processdata.py). ETL pipeline loads data, merges and cleans it and saves to database file.
Database file DisasterResponse.db.

To run ETL pipeline:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

- Models folder 
Machine learning pipeline script (train_classifier.py). It loads data from database file, builds and trains classifier and saves model.
Trained model classifier.pickle.

To run ML pipeline:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


- App folder 
run.py file runs the webapp
Template folder contains html files master.html (visualisation) and go.html (model work demonstration).

To run web app:

python run.py