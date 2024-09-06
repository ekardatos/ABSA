# Import Libraries
from train import model_train
from test import model_test

# The split xml files
all_data = ['XML Files/part1.xml', 'XML Files/part2.xml', 'XML Files/part3.xml', 'XML Files/part4.xml', 'XML Files/part5.xml',
            'XML Files/part6.xml', 'XML Files/part7.xml', 'XML Files/part8.xml', 'XML Files/part9.xml', 'XML Files/part10.xml']

# Score
accuracy_scores = []

print('Running a 10-fold cross validation experiment with a 9/1 split')
for index, file in enumerate(all_data):
    print(f"10-fold cross validation using {file} as the test file")

    # Use .copy() funtion to create a copy of the old list because otherwise i pop elements from the all_data list and i
    # alter the for loop
    train_filenames = all_data.copy()
    train_filenames.pop(index)

    # Call the model_train function with the train_filenames as files
    model_train(train_filenames=train_filenames)

    # Use as a test file only the current file of the for loop
    test_filenames = [file]

    # Call the model_test function with the test file
    accuracy_scores.append(model_test(test_filenames=test_filenames))

print("Average Accuracy Score is: " + str(sum(accuracy_scores)/len(accuracy_scores)))
