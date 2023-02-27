import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm


def creat_svm_dataset(X_file, sp=256):
    X_raw = np.load(X_file, mmap_mode='r+')
    dl = DataLoader(X_raw)
    d = iter(dl)
    X = []

    while True:
        try:
            t = np.array(next(d))[0]
            l = []
            t1 = np.split(t, sp)
            for i in t1:
                t2 = np.split(i, sp, 1)
                for j in t2:
                    r = j[:, :, 0].mean()
                    g = j[:, :, 1].mean()
                    b = j[:, :, 2].mean()

                    l += [0.1 * r + 0.6 * g + 0.3 * b]
            X += [l]
        except:
            break
    X = np.array(X).round(1)
    return X


# apply the above function
def accuracy_svm():
    X = creat_svm_dataset('X_train.npy')
    y = np.load('y_train.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    svm_model = svm.SVC(decision_function_shape='ovr')
    svm_model.fit(X_train, y_train)
    svm_y_pred = svm_model.predict(X_test)
    a = accuracy_score(y_test, svm_y_pred)
    print(f"accuracy score:{a}")


def predict_y_test():
    X = creat_svm_dataset('X_train.npy')
    y = np.load('y_train.npy')
    svm_model = svm.SVC(decision_function_shape='ovr')
    svm_model.fit(X, y)
    X_test = creat_svm_dataset('X_test.npy')
    svm_y_pred = svm_model.predict(X_test)
    print(f"y_predict:{svm_y_pred}")
    np.save('y_test_of_SVM.npy',svm_y_pred)


# accuracy_svm()
# predict_y_test()