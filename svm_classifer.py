import pickle

from sklearn import svm


def train(X, y):
    C = 1.0  # SVM regularization parameter
    model = svm.LinearSVC(C=C, max_iter=10000)
    model = model.fit()
    return model


def save_model(model):
    pickle.dump(model, open("saved_svm.sav", "wb"))


def load_model(file_path):
    loaded_model = pickle.load(open(file_path, "rb"))
    return loaded_model


def predict(model, X):
    y = model.predict(X)
    return y
