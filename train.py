# Import libraries
import pandas as pd
from joblib import dump
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from load import load_file

# The split xml files
all_filenames = {'XML Files/part1.xml', 'XML Files/part2.xml', 'XML Files/part3.xml', 'XML Files/part4.xml', 'XML Files/part5.xml',
                 'XML Files/part6.xml', 'XML Files/part7.xml', 'XML Files/part8.xml', 'XML Files/part9.xml', 'XML Files/part10.xml'}

# Function that trains the model
def model_train(train_filenames):
    print('-' * 40)
    print('Running model_train function')
    train_filenames = set(train_filenames)

    # Putting all data into train_data
    train_data = pd.DataFrame()

    # Loading every single file and putting into train_data
    for train_filename in train_filenames:
        data_file = load_file(train_filename)
        # Insert every file in the train_data
        train_data = pd.concat([train_data, data_file])

    # Calculate the number of train_data to use it
    # After that, the TfifdVectorizer to obtain just the train data
    number_of_train_data = len(train_data)

    # Use the all_data variable to load all the data and used by the TfidfVectorizer
    all_data = train_data

    # Loading every single file and putting into train_data
    for filename in (all_filenames - train_filenames):
        data_file = load_file(filename)
        # Insert every file in the train_data
        all_data = pd.concat([all_data, data_file])

    # Use 'Sentence' from our data to train model
    text = all_data['Sentence']

    # Tf-Idf for n-grams in word level
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        # From 2 to 6 n-grams
        ngram_range=(2, 6),
        max_features=50000)
    # Fit
    word_vectorizer.fit(text)

    # Transform in order to get the vector representations
    train_word_features = word_vectorizer.transform(text)

    # All that is transformed will be used for training putting them in a horizontal order (hstack)
    # Use the number_of_train_data in order to obtain the first n rows as those rows are the train data
    train_features = train_word_features[:number_of_train_data]

    # The target is the Class column of the data
    train_target = train_data['Class']

    # Use a Logistic Regression model
    classifier = LogisticRegression()

    # X_train -- train_features, y_train -- train_target
    classifier.fit(train_features, train_target)

    # Score to compute the accuracy
    print(f"Train score is: {classifier.score(train_features, train_target)}")
    print('-' * 40)
    print()

    # Save the model
    model_name = "Logistic_Regr_model.joblib"
    dump(classifier, model_name)

# This allows our program to be executable by itself
if __name__ == '__main__':
    # The train xml files
    tf = ['XML Files/part1.xml', 'XML Files/part2.xml', 'XML Files/part3.xml', 'XML Files/part4.xml', 'XML Files/part5.xml',
          'XML Files/part6.xml', 'XML Files/part7.xml']
    print(tf)
    model_train(tf)