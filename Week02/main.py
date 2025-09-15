import sys
import argparse
import os
import pickle
from model import Model, Dataset
from scikit-learn import svm.SVC as SVC

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to the model file")
parser.add_argument("--input", type=str, required=True, help="Path to the input data file")
def main():
    args = parser.parse_args()
    model_path = args.model
    input_path = args.input

    dataset = Dataset(data_path=input_path)
    texts, labels = dataset.load_data()
    texts_preprocessed = dataset.preprocess(texts)
    X_train, X_test, y_train, y_test = dataset.split(texts_preprocessed, labels)

    model = Model(model=SVC())
    model.train(texts_preprocessed, labels)
    accuracy, report = model.evaluate(texts_preprocessed, labels)

    print(f"Accuracy: {accuracy}")



