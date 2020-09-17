# Disaster Response Pipeline Project

### Project description:

- Data folder contains source csv file and script for ETL pipeline (processdata.py). ETL pipeline loads data, merges and cleans it and saves to database file.

To run ETL pipeline:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

- Models folder contains machine learning pipeline script (train_classifier.py). It loads data from database file, builds and trains classifier and saves trained model to pickle file.

To run ML pipeline:

python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


- App folder contains files for webapp.

To run web app:

python run.py
