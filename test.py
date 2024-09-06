# Import libraries
import pandas as pd
from joblib import load
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from load import load_file

# The split xml files
all_filenames = {'XML Files/part1.xml', 'XML Files/part2.xml', 'XML Files/part3.xml', 'XML Files/part4.xml', 'XML Files/part5.xml',
                 'XML Files/part6.xml', 'XML Files/part7.xml', 'XML Files/part8.xml', 'XML Files/part9.xml', 'XML Files/part10.xml'}

# The model_test function as in train
def model_test(test_filenames):
    print('-' * 40)
    print('Running model_test function')
    # The set containing the parts that we are going to use for the test data
    test_filenames = set(test_filenames)

    # Initialize the Test Data
    test_data = pd.DataFrame()

    # Load every single file and putting into test_data
    for test_filename in test_filenames:
        data_file = load_file(test_filename)
        # Insert every file in the test_data
        test_data = pd.concat([test_data, data_file])

    # We calculate the number of train_data to use it
    # After that the TfifdVectorizer to obtain just the train data
    number_of_test_data = len(test_data)

    # Use the all_data variable to load all the data and used by the TfidfVectorizer
    all_data = test_data

    # Load every single file and putting into train_data
    for filename in (all_filenames - test_filenames):
        data_file = load_file(filename)
        # Insert every file in the train_data
        all_data = pd.concat([all_data, data_file])

    # Use only the sentence from our data to train model
    text = all_data['Sentence']

    # Tf-Idf for n-grams in word level
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        # From 2 to 6 n-grams
        ngram_range=(2, 6),
        max_features=50000)
    word_vectorizer.fit(text)

    # Use the word_vectorizer to transform the text to vector representation for the n-grams
    test_word_features = word_vectorizer.transform(text)

    # Use horizontal stack in the test_word_features and test_word_features to create the full feature set
    # Use the number_of_test_data in order to obtain the first n rows as those rows are the test data
    test_features = test_word_features[:number_of_test_data]

    # Load the model classifier that was created in the training phase
    classifier = load('Logistic_Regr_model.joblib')

    # The target is the Class column of the data
    test_target = test_data['Class']

    # Score to compute the accuracy
    score = classifier.score(test_features, test_target)
    print(f"Test score is: {score}")
    print('-' * 40)
    return score

#This allows our program to be executable by itself
if __name__ == '__main__':
    # The test xml files
    tf = ['XML Files/part8.xml', 'XML Files/part9.xml', 'XML Files/part10.xml']
    print(tf)
    model_test(tf)