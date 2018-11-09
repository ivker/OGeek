from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib


class RFRegressor:

    def __init__(self, encoder=None):
        self.clf = RandomForestRegressor(n_estimators=10, criterion="mse", max_depth=None, min_samples_split=2,
                                         min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto",
                                         max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None,
                                         bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
                                         verbose=0, warm_start=False)

        self.encoder = encoder

    def fit(self, x, y):
        if self.encoder is not None:
            x = self.encoder.oneHot(x, new='add')
        self.clf.fit(x, y)

    def predict(self, x):
        if self.encoder is not None:
            x = self.encoder.oneHot(x, new='none')
        return self.clf.predict(x)

    def dump(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
