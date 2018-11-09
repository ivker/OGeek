from sklearn.externals import joblib


class Model:
    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def dump(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
