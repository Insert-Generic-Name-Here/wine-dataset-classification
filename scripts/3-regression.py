## Importing Essential Libraries and Modules
import os
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

### Defining Essential Functions
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    import itertools
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def quality_labels(y):
    if y <= 4:
        return -1
    elif y <= 6:
        return 0
    else:
        return 1


def regression_red_wine(rn_wine_X, rn_wine_y):
    ### * Vinho Verde Red Wine; Normalized; Outlier-Free
    #### Having the Quality Labels as is (For Regression)
    #### Having the Quality Labels Separated into 3 Classes (For Classification)
    #### * Bad (<=4), Mediocre (<=6), Good (<=10)
    rn_wine_y_class = np.array(list(map(quality_labels, rn_wine_y)))


    ## Cross Validation
    #### Use K-Fold Cross Validation to split the Dataset into 5 random partitions 
    #### (test with one; train with all others) and repeat for 5 epochs
    from tqdm import tqdm
    from sklearn.model_selection import RepeatedKFold
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=100)

    accuracy_lr = []
    accuracy_knn = []

    ## Logistic Regression
    ## * View the accuracy of regression having all features predict the rating value
    regressor = LogisticRegression(solver='newton-cg', n_jobs=-1)
    accuracy = np.array([])

    for train_index, test_index in tqdm(rkf.split(rn_wine_X), total=rkf.get_n_splits()):
        x_train, x_test = rn_wine_X[train_index], rn_wine_X[test_index]
        y_train, y_test = rn_wine_y[train_index], rn_wine_y[test_index]
        regressor.fit(x_train,y_train)
        
        accuracy = np.append(accuracy, [regressor.score(x_test, y_test)])

    chi = np.abs(accuracy).mean() * 100
    accuracy_lr.append(chi)
    print ("Accuracy (Logistic Regression; All Features): {}%".format(chi))


    ## k-Nearest Neighbors
    ## * View the accuracy of regression having all features predict the rating value
    knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='minkowski', n_jobs=-1)
    accuracy = np.array([])

    for train_index, test_index in tqdm(rkf.split(rn_wine_X), total=rkf.get_n_splits()):
        x_train, x_test = rn_wine_X[train_index], rn_wine_X[test_index]
        y_train, y_test = rn_wine_y_class[train_index], rn_wine_y_class[test_index]
        knn_classifier.fit(x_train,y_train)
        accuracy = np.append(accuracy, [knn_classifier.score(x_test, y_test)])

    chi = np.abs(accuracy).mean() * 100
    accuracy_knn.append(chi)
    print ("Accuracy (K-Nearest Neignbors; All Features): {}%".format(chi))


    ##  (Theoretically) Increasing Prediction Accuracy Using Feature Selection
    #### By using Feature Selection we (theoretically):
    #### * Reduce Overfitting
    #### * Improve the Accuracy of the used Predictor
    #### * Reduces the Training Time (as we now train on a portion of the samples' features)


    ### Training with 5 features (SelectKBest Result)
    ### * Transform the Dataset; Train; Compute Accuracy and Confusion Matrices
    ### * Logistic Regression
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2

    # Feature Extraction
    test = SelectKBest(score_func=chi2, k=5)
    fit = test.fit(rn_wine_X, rn_wine_y)

    # Transform the Dataset
    rn_wine_X5 = fit.transform(rn_wine_X)

    regressor = LogisticRegression(solver='newton-cg', n_jobs=-1)
    accuracy = np.array([])

    for train_index, test_index in tqdm(rkf.split(rn_wine_X5), total=rkf.get_n_splits()):
        x_train, x_test = rn_wine_X5[train_index], rn_wine_X5[test_index]
        y_train, y_test = rn_wine_y[train_index], rn_wine_y[test_index]
        regressor.fit(x_train,y_train)
        accuracy = np.append(accuracy, [regressor.score(x_test,y_test)])

    chi = np.abs(accuracy).mean() * 100
    accuracy_lr.append(chi)
    print ("Accuracy (Logistic Regression; 5 Most Important Features): {}%".format(chi))

    knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2, metric='minkowski', n_jobs=-1)
    accuracy = np.array([])

    for train_index, test_index in tqdm(rkf.split(rn_wine_X5), total=rkf.get_n_splits()):
        x_train, x_test = rn_wine_X5[train_index], rn_wine_X5[test_index]
        y_train, y_test = rn_wine_y_class[train_index], rn_wine_y_class[test_index]
        knn_classifier.fit(x_train,y_train)
        accuracy = np.append(accuracy, [knn_classifier.score(x_test,y_test)])

    chi = np.abs(accuracy).mean() * 100
    accuracy_knn.append(chi)
    print ("Accuracy (K-Nearest Neighbors; 5 Most Important Features): {}%".format(chi))


    ### Training with 4 features (ExtraTreesClassifier Result)
    ### * Transform the Dataset; Train; Compute Accuracy and Confusion Matrices
    ### * Logistic Regression
    usefulFeatures = [False, True, False, False, False, False, True, False, False, True, True]
    rn_wine_X4 = rn_wine_X[:,usefulFeatures]

    regressor = LogisticRegression(solver='newton-cg', n_jobs=-1)
    accuracy = np.array([])

    for train_index, test_index in tqdm(rkf.split(rn_wine_X4), total=rkf.get_n_splits()):
        x_train, x_test = rn_wine_X4[train_index], rn_wine_X4[test_index]
        y_train, y_test = rn_wine_y[train_index], rn_wine_y[test_index]
        regressor.fit(x_train,y_train)
        accuracy = np.append(accuracy, [regressor.score(x_test,y_test)])

    chi = np.abs(accuracy).mean() * 100
    accuracy_lr.append(chi)
    print ("Accuracy (Logistic Regression; 4 Most Important Features): {}%".format(chi))

    knn_classifier = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2, metric='minkowski', n_jobs=-1)
    accuracy = np.array([])

    for train_index, test_index in tqdm(rkf.split(rn_wine_X4), total=rkf.get_n_splits()):
        x_train, x_test = rn_wine_X4[train_index], rn_wine_X4[test_index]
        y_train, y_test = rn_wine_y_class[train_index], rn_wine_y_class[test_index]
        knn_classifier.fit(x_train,y_train)
        accuracy = np.append(accuracy, [knn_classifier.score(x_test,y_test)])

    chi = np.abs(accuracy).mean() * 100
    accuracy_knn.append(chi)
    print ("Accuracy (K-Nearest Neighbors; 4 Most Important Features): {}%".format(chi))

    return accuracy_lr, accuracy_knn

if __name__ == '__main__':
    ### Defining Cache Directories
    np_cache_dir = os.path.join('..', 'numpy_cache')
    csv_cache_dir = os.path.join('..', 'csv_cache')

    rn_wine_X = np.load(os.path.join(np_cache_dir, 'red_clean_X_normed.npy'))
    rn_wine_y = np.load(os.path.join(np_cache_dir, 'red_clean_y.npy'))

    regression_red_wine(rn_wine_X, rn_wine_y)