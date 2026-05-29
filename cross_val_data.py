import mnist_reader
import numpy as np
from sklearn.model_selection import KFold

def cross_validation() -> list:
    """
    return 5 disjoint folders of dataset
    use to conduct validation
    """
    X_train_all, y_train_all = mnist_reader.load_mnist('./data/fashion', kind='train')

    X_train_all = X_train_all / 255.0

    kfold = KFold(n_splits=5, shuffle=True, random_state=178)
    tr_fold = []    # list contains X_tr, y_tr      ex [[X_tr_1, y_tr_1], [X_tr_2, y_tr_2]...]
    val_fold = []   # simliar as tr_fold

    for tr_idx, val_idx in kfold.split(X_train_all):
        X_tr = X_train_all[tr_idx].reshape(-1,28,28,1)
        y_tr = y_train_all[tr_idx]

        X_val = X_train_all[val_idx].reshape(-1,28,28,1)
        y_val = y_train_all[val_idx]

        tr_fold.append([X_tr, y_tr])
        val_fold.append([X_val, y_val])
    
    return tr_fold, val_fold


if __name__ == "__main__":
    X_tr, y_tr = cross_validation()
    print(X_tr[0][0].shape)
    print(y_tr[0][0].shape)