
import numpy as np
import mnist_reader
from sklearn.model_selection import train_test_split

def get_data(val_size=0.2):
    # load data
    X_train_all, y_train_all = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

    # normalize data
    X_train_all = X_train_all / 255.0
    X_test = X_test / 255.0

    # split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=val_size,
        random_state=3,
        stratify=y_train_all
    )

    return X_train, X_val, X_test, y_train, y_val, y_test