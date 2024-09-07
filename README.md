# Aspect-Based Sentiment Analysis (ABSA)

This project implements an Aspect-Based Sentiment Analysis (ABSA) system for mining and summarizing opinions from text about specific entities, such as a restaurant. The system is built to detect the main aspects discussed in the text and estimate the polarity (positive, neutral, or negative) for each aspect.

## Dataset

The dataset used in this project comes from the [SemEval-2016 ABSA Restaurant Reviews](http://metashare.ilsp.gr:8080/repository/browse/semeval-2016-absa-restaurant-reviews-english-train-data-subtask-1/cd28e738562f11e59e2c842b2b6a04d703f9dae461bb4816a5d4320019407d23/). It consists of 350 restaurant reviews split into 2000 sentences. Each sentence has been manually annotated with aspects (`category` attribute) and assigned a `polarity` label (positive, neutral, or negative).

## Project Structure

The project includes the following files:

- **`split.py`**: A Python script to parse the XML dataset and split it into 10 parts (35 reviews per part). Each part is saved as a separate XML file (`part1.xml`, `part2.xml`, …, `part10.xml`).
  
- **`train.py`**: Contains a function to train a machine learning model using a specified set of parts from the dataset. The model is saved to disk after training. The function supports Logistic Regression, and utilizes a variety of features (e.g., unigrams, bigrams, trigrams).

- **`test.py`**: Includes a function to load a pre-trained model and predict the polarities for the aspects of a specified part of the dataset.

- **`experiments.py`**: Performs 10-fold cross-validation using the functions from `train.py` and `test.py`. At each iteration, 9 parts are used for training and 1 part for testing/evaluation. Accuracy is calculated for each fold and the average accuracy is computed.

