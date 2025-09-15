from sklearn import [
    naive_bayes.MultinomialNB,
    svm.SVC,
    linear_model.LogisticRegression,
    model_selection.train_test_split,
    feature_extraction.text.CountVectorizer,
    feature_extraction.text.TfidfTransformer,
    metrics.classification_report,
    metrics.accuracy_score
]
import pandas as pd
import pickle

COMMON_TEXT_NAME = ['text', 'review', 'content', 'comment', 'post']
COMMON_LABEL_NAME = ['label', 'target', 'class', 'sentiment']

class Model():
    def __init__(self, model)
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

class Dataset():
    def __init__(self, data_path, vectorizer=None, tfidf_transformer=None):
        self.data_path = data_path
        if vectorizer is None:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = vectorizer
        if tfidf_transformer is None:
            self.tfidf_transformer = TfidfTransformer()
        else:
            self.tfidf_transformer = tfidf_transformer

    def load_data(self):
        df = pd.read_csv(self.data_path)
        for col in df.columns:
            if col.lower() in COMMON_TEXT_NAME:
                text_col = col
            if col.lower() in COMMON_LABEL_NAME:
                label_col = col
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].tolist()
        return texts, labels

    def preprocess(self, texts):
        X_counts = self.vectorizer.fit_transform(texts)
        X_tfidf = self.tfidf_transformer.fit_transform(X_counts)
        return

    def split(self, texts, labels, test_size=0.2, random_state=42):
        return train_test_split(texts, labels, test_size=test_size, random_state=random_state)