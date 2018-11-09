import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


class RFClassifier:

    def __init__(self, encoder=None, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1,
                 random_state=None, verbose=0, warm_start=False, class_weight=None):
        self.clf = RandomForestClassifier(n_estimators, criterion, max_depth, min_samples_split,
                                          min_samples_leaf, min_weight_fraction_leaf,
                                          max_features, max_leaf_nodes, min_impurity_decrease,
                                          min_impurity_split, bootstrap, oob_score, n_jobs,
                                          random_state, verbose, warm_start, class_weight)

        self.encoder = encoder

    def fit(self, x, y):
        if self.encoder is not None:
            x = np.asarray(self.encoder.oneHot(x, new='add'), dtype=np.int32)
        self.clf.fit(x, np.array(y).astype('str'))

    def predict(self, x):
        if self.encoder is not None:
            x = self.encoder.oneHot(x, new='none')
        return self.clf.predict(x)

    def dump(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
