import numpy as np

from cn.ivker.ogeek.core.preprocess.Regress import Regress
from cn.ivker.ogeek.util.Array import Array


class OppClassifiedData:

    def __init__(self):
        self.train_x = None
        self.train_x_dict = None
        self.train_y = None
        self.valid_x = None
        self.valid_x_dict = None
        self.valid_y = None
        self.test_x = None
        self.test_x_dict = None

    def fitTrain(self, prefix, predict, title, tag, label):
        """
        准备数据,并不是所有准备的数据都会用于学习
        :param prefix:
        :param predict:
        :param title:
        :param tag:
        :param label:
        :return:
        """
        # 最大预测输入
        max_predict = self._maxPredict(predict)

        # *.拼接特征
        self.train_x = self._feature2vector([title, tag, max_predict])

        self.train_y = np.asarray(label)

        return [self.train_x, self.train_y]

    def fitValid(self, prefix, predict, title, tag, label):
        """
        返回特征+点击率(回归)
        :param prefix:
        :param predict:
        :param title:
        :param tag:
        :param label:
        :return:
        """
        # 最大预测输入
        max_predict = self._maxPredict(predict)

        # *.拼接特征
        self.valid_x = self._feature2vector([title, tag, max_predict])

        self.valid_y = np.asarray(label)

        return [self.valid_x, self.valid_y]

    def fitTest(self, prefix, predict, title, tag):
        # 最大预测输入
        max_predict = self._maxPredict(predict)

        # *.拼接特征
        self.test_x = self._feature2vector([title, tag, max_predict])

        return self.test_x

    def getTrain(self, extract=None):
        """
        获取训练数据,当extract不为空时,获取局部训练数据
        :param extract: 局部数据索引
        :return:
        """
        if extract is not None:
            x = Array.extract(self.train_x, extract)
            y = Array.extract(self.train_y, extract)
            [x, y] = Regress.binaryClassified2Regress(x, y)
        else:
            [x, y] = Regress.binaryClassified2Regress(self.train_x, self.train_y, self.train_x_dict)

        # *.将分类问题转为回归值

        return [x, y]

    def getValid(self, extract=None):
        """
        返回特征+点击率(回归)
        :param extract:
        :return:
        """
        if extract is not None:
            x = Array.extract(self.valid_x, extract)
            y = Array.extract(self.valid_y, extract)
        else:
            x = self.valid_x
            y = self.valid_y

        return [x, y]

    def getTest(self, extract=None):
        if extract is not None:
            x = Array.extract(self.test_x, extract)
        else:
            x = self.test_x

        return x

    @staticmethod
    def _maxPredict(records_predicts):
        """

        :param records_predicts:
        :return:
        """
        result = []
        for predicts in records_predicts:
            max_predict = "0"
            max_predict_value = 0
            for key in predicts:
                if predicts[key] > max_predict_value:
                    max_predict = key
                    max_predict_value = predicts[key]
            result.append(max_predict)

        return result

    @staticmethod
    def _feature2vector(f_list: list):
        """
        将数据拼接成特征向量
        :param f_list:每个特征list中可能
        :return:
        """

        return np.asarray(f_list).T
