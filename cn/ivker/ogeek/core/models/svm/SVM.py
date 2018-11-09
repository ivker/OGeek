from sklearn.externals import joblib
from sklearn.svm import SVC

from cn.ivker.ogeek.core.models.Model import Model


class SVM(Model):

    def __init__(self, encoder=None):
        self.clf = SVC(C=1.0, cache_size=4000, class_weight=None, coef0=0.0,
                       decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                       max_iter=-1, probability=False, random_state=None, shrinking=True,
                       tol=0.001, verbose=False)
        self.encoder = encoder

    def fit(self, x, y):
        if self.encoder is not None:
            x = self.encoder.oneHot(x, new='add')
        self.clf.fit(x, y)

    def predict(self, x):
        if self.encoder is not None:
            x = self.encoder.oneHot(x, new='none')
        return self.clf.predict(x)




