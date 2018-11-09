import jieba as jb
import numpy as np
import logging
from cn.ivker.ogeek.core.preprocess.Regress import Regress
from cn.ivker.ogeek.core.statistics.BinaryClassifierStatistician import BinaryClassifierStatistician
from cn.ivker.ogeek.util.Array import Array
from cn.ivker.ogeek.util.Distance import Distance

logger = logging.getLogger("OppRegressedData")


class OppRegressedData:

    def __init__(self):
        self.train_x = None
        self.train_x_dict = None
        self.train_y = None
        self.valid_x = None
        self.valid_x_dict = None
        self.valid_y = None
        self.test_x = None
        self.test_x_dict = None

        self.predict_tag_dict = None
        self.predict_title_stat = None

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
        # F1:tag
        # F2:prefix和title编辑距离
        # F3:max_predict和title编辑距离
        # F4/5:max_predict分词,与tag的正/负样本个数
        # F6/7:max_predict分词,与title的正/负样本个数
        max_predict = self._maxPredict(predict, prefix)

        # 编辑距离
        # prefix_title_ed = list(map(lambda x: Distance.edit_distance(x[0], x[1]), zip(prefix, title)))
        prefix_title_ed = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(prefix, title)))

        # max_predict_title_ed = list(map(lambda x: Distance.edit_distance(x[0], x[1]), zip(max_predict, title)))
        max_predict_title_ed = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(max_predict, title)))

        # 分词
        # predict_words = \
        #     list(map(lambda x: list(filter(lambda word: word not in [""], jb.cut(x, cut_all=True))), max_predict))
        # title_words = list(map(lambda x: list(filter(lambda word: word not in [""], jb.cut(x, cut_all=True))), title))
        #
        # # 统计正负样本数
        # new_x = self._fitWordsRelatedTagStatistician(predict_words, tag, label)
        # [predict_tag_p, predict_tag_n] = self._getWordsRelatedTagStatistician(new_x)
        # self.train_x = self._feature2vector([tag, prefix_title_ed, max_predict_title_ed, predict_tag_p, predict_tag_n])
        self.train_x = self._feature2vector([tag, prefix_title_ed, max_predict_title_ed])
        self.train_y = np.asarray(label)

        return [self.train_x, self.train_y]

    def fitValid(self, prefix, predict, title, tag, label):
        max_predict = self._maxPredict(predict, prefix)

        # 编辑距离
        # prefix_title_ed = list(map(lambda x: Distance.edit_distance(x[0], x[1]), zip(prefix, title)))
        prefix_title_ed = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(prefix, title)))
        # max_predict_title_ed = list(map(lambda x: Distance.edit_distance(x[0], x[1]), zip(max_predict, title)))
        max_predict_title_ed = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(max_predict, title)))

        # 分词
        # predict_words = \
        #     list(map(lambda x: list(filter(lambda word: word not in [""], jb.cut(x, cut_all=True))), max_predict))

        # self.valid_x = self._feature2vector([tag, max_predict_title_ed])
        # new_x = self._wordsRelatedTag(predict_words, tag)
        # [predict_tag_p, predict_tag_n] = self._getWordsRelatedTagStatistician(new_x)
        # self.valid_x = self._feature2vector([tag, prefix_title_ed, max_predict_title_ed,predict_tag_p, predict_tag_n])
        self.valid_x = self._feature2vector([tag, prefix_title_ed, max_predict_title_ed])
        self.valid_y = np.asarray(label)

        return [self.valid_x, self.valid_y]

    def fitTest(self, prefix, predict, title, tag):
        max_predict = self._maxPredict(predict, prefix)

        # 编辑距离
        # prefix_title_ed = list(map(lambda x: Distance.edit_distance(x[0], x[1]), zip(prefix, title)))
        prefix_title_ed = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(prefix, title)))
        # max_predict_title_ed = list(map(lambda x: Distance.edit_distance(x[0], x[1]), zip(max_predict, title)))
        max_predict_title_ed = list(map(lambda x: 1 if x[0] == x[1] else 0, zip(max_predict, title)))

        self.test_x = self._feature2vector([tag, prefix_title_ed, max_predict_title_ed])

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

        return [x, y]

    def getValid(self, extract=None):
        """
        返回特征
        :param extract:
        :return:
        """
        if extract is not None:
            x = Array.extract(self.valid_x, extract)
            y = Array.extract(self.valid_y, extract)
        else:
            [x, y] = [self.valid_x, self.valid_y]

        return [x, y]

    def getTest(self, extract=None):
        """
        返回特征
        :param extract:
        :return:
        """
        if extract is not None:
            x = Array.extract(self.test_x, extract)
        else:
            x = self.test_x

        return x

    @staticmethod
    def _feature2vector(f_list: list):
        """
        将数据拼接成特征向量
        :param f_list:每个特征list中可能
        :return:
        """

        return np.asarray(f_list).T

    @staticmethod
    def _maxPredict(records_predicts, prefix):
        """

        :param records_predicts:
        :return:
        """
        result = []
        for idx in range(len(prefix)):
            predicts = records_predicts[idx]
            max_predict = prefix[idx]
            max_predict_value = 0
            for key in predicts:
                if predicts[key] > max_predict_value:
                    max_predict = key
                    max_predict_value = predicts[key]
            result.append(max_predict)

        return result

    def _fitWordsRelatedTagStatistician(self, words, tag, label):
        """
        将每条记录中的分词统计为二分类样本
        :param words:
        :param tag:
        :param label:
        :return:
        """
        new_x = []
        self.predict_tag_dict = {}
        r_size = len(label)
        for idx in range(r_size):
            un = list(set(words[idx]))
            t = []
            for w in un:
                key = w + tag[idx]
                t.append(key)
                if self.predict_tag_dict.get(key) is not None:
                    if label[idx] in [1, '1']:
                        self.predict_tag_dict[key][0] += 1
                    else:
                        self.predict_tag_dict[key][1] += 1
                else:
                    if label[idx] in [1, '1']:
                        self.predict_tag_dict.setdefault(key, [1, 0])
                    else:
                        self.predict_tag_dict.setdefault(key, [0, 1])
            new_x.append(t)

        # 将统计量按500为单位取整数
        for key in self.predict_tag_dict:
            self.predict_tag_dict[key][0] = int(self.predict_tag_dict[key][0] / 500)
            self.predict_tag_dict[key][1] = int(self.predict_tag_dict[key][1] / 500)

        logger.info("statistics,fit后的规模为%d" % len(self.predict_tag_dict))

        return new_x

    def _getWordsRelatedTagStatistician(self, wt):
        p = []
        n = []
        statistics = self.predict_tag_dict
        for record in wt:
            pc = 0
            nc = 0
            for key in record:
                st = statistics.get(key)
                if st is not None:
                    pc += st[0]
                    nc += st[1]
            p.append(pc)
            n.append(nc)
        return [p, n]

    @staticmethod
    def _wordsRelatedTag(words, tag):
        title_tag = []
        r_size = len(words)
        for idx in range(r_size):
            un = list(set(words[idx]))
            l = []
            for w in un:
                l.append(w + tag[idx])
            title_tag.append(l)
        # 添加历史记录
        return title_tag

    @staticmethod
    def _wordsRelatedWordsStatistician(words1, words2, label):
        """
        将每条记录中的分词统计为二分类样本
        :param words1:
        :param words2:
        :param label:
        :return:
        """
        title_tag = []
        title_tag_label = []
        related = BinaryClassifierStatistician()
        r_size = len(label)
        for idx in range(r_size):
            un1 = list(set(words1[idx]))
            un2 = list(set(words2[idx]))
            for w1 in un1:
                for w2 in un2:
                    title_tag.append(w1 + w2)
                    title_tag_label.append(label[idx])

        # 添加历史记录
        related.fit(title_tag, title_tag_label)
        return related

    @staticmethod
    def word2FeatureVector(records: list, feature_mapping: dict):
        """
        将词特征转为特征向量
        :param records:
        :param feature_mapping:
        :return:
        """
        shape_x = len(records)
        shape_y = len(feature_mapping)
        vector = np.zeros((shape_x, shape_y), np.int8)
        for idx in range(shape_x):
            words = records[idx]
            for word in words:
                feature_id = feature_mapping.get(word)
                if feature_id is not None:
                    vector[idx][feature_id] = 1
        return vector
