import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
import scikitplot as skplt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def evaluate_features(X, y, clfKey):
    """
    General helper function for evaluating effectiveness of passed features in ML model

    Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation

    Parameters
    ----------
    X : Features array

    y : Labels array

    clfKey: 'LogReg', 'SDG_Mh', 'SDG_Log' and 'Tree'
    """

    clf = {
        'LogReg': LogisticRegression(),
        'SDG_Mh': SGDClassifier(loss="modified_huber", penalty="l2", max_iter=5),
        'SDG_Log': SGDClassifier(loss="log", penalty="l2", max_iter=5),
        'Tree': RandomForestClassifier(n_estimators=1000, max_depth=5, verbose=1)
    }

    probabilities = cross_val_predict(clf[clfKey], X, y, cv=StratifiedKFold(n_splits=2, random_state=8),
                                      n_jobs=-1, method='predict_proba', verbose=2)
    predicted_indices = np.argmax(probabilities, axis=1)
    classes = np.unique(y)
    predicted = classes[predicted_indices]
    print('Log loss: {}'.format(log_loss(y, probabilities)))
    print('Accuracy: {}'.format(accuracy_score(y, predicted)))
    skplt.metrics.plot_confusion_matrix(y, predicted, normalize=True)
    plt.show()


""" 
Information Input:
    This section reads in the Training information in and then attaches it to each other it will store it as a csv 
    and hold all of the information of a row together. 
"""
training_text = pd.read_csv('./training_text', sep='\|\|', header=None, skiprows=1, names=["ID", "Text"],
                            engine='python')

train_variants = pd.read_csv('./training_variants')

trainingDataSet = pd.merge(train_variants, training_text, how='left', on='ID')

# print(trainingDataSet.head())


"""
Build: 
    Count Vectorizer:
    
    TFIDF Vectorizer:   
"""
print(type(trainingDataSet['Text']))

x = False
# --------------------------------------- Count Vectorizer (CV) --------------------------------------------------------
if x:
    CV = CountVectorizer(
        analyzer='word',
        strip_accents='unicode',
        # stop_words='english',
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b',
        min_df=1
    )

    CV = CV.fit_transform(trainingDataSet['Text'].values.astype('U'))

    CV_SDV = TruncatedSVD(n_components=100, n_iter=5, random_state=12)

    CV_SDV = CV_SDV.fit_transform(CV)

# ------------------------------------- TFIDF Vectorizer (TFIDF) -------------------------------------------------------

if x:
    TFIDF = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b',
        min_df=1,
        strip_accents='unicode',
        sublinear_tf=True
    )

    TFIDF = TFIDF.fit_transform(trainingDataSet['Text'].values.astype('U'))

    TFIDF_SDV = TruncatedSVD(n_components=100, n_iter=5, random_state=12)

    TFIDF_SDV = TFIDF_SDV.fit_transform(TFIDF)

"""
Evaluation
"""
# evaluate_features(
#     TFIDF_SDV,
#     trainingDataSet['Class'].values.ravel(),
#     'LogReg'  # clfKey: 'LogReg', 'SDG_Log', 'SDG_Mh' and 'Tree'
# )
