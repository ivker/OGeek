from cn.ivker.ogeek.core.preprocess.OppDataFormat import OppDataFormat
from cn.ivker.ogeek.core.statistics.BinaryClassifierStatistician import BinaryClassifierStatistician
from cn.ivker.ogeek.util.File import File

import synonyms as syn
import numpy as np
import matplotlib.pyplot as plt
import jieba as jb

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


def sameTest():
    _train_path = "../../dataset/oppo_round1_train_20180929/oppo_round1_train_20180929.txt"
    train_records = File.readList(_train_path)
    [prefix, predict, title, tag, label] = OppDataFormat.withResult(train_records)
    max_predict = _maxPredict(predict)

    count = 0
    positive = 0
    negative = 0

    r_size = len(label)
    for idx in range(r_size):
        if prefix[idx] == title[idx] and max_predict[idx] == title[idx]:
            count += 1
            if label[idx] in [1, "1"]:
                positive += 1
            else:
                negative += 1

    print("same_count:%d,positive:%d,negative:%d" % (count, positive, negative))


def similarTest():
    _train_path = "../../dataset/oppo_round1_train_20180929/oppo_round1_train_20180929.txt"
    valid_path = "../../dataset/oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt"
    tmp_path = "../../dataset/test.txt"
    train_records = File.readList(valid_path)
    [prefix, predict, title, tag, label] = OppDataFormat.withResult(train_records)
    max_predict = _maxPredict(predict)

    r_size = len(label)
    similar = np.zeros(r_size)
    p_count = {}
    n_count = {}
    for idx in range(r_size):
        similar = syn.compare(max_predict[idx], title[idx])
        if label[idx] in [1, "1"]:
            if p_count.get(similar) is None:
                p_count.setdefault(similar, 1)
            else:
                p_count[similar] += 1
        else:
            if n_count.get(similar) is None:
                n_count.setdefault(similar, -1)
            else:
                n_count[similar] -= 1
    plt.plot(p_count.keys(), p_count.values(), "*")
    plt.plot(n_count.keys(), n_count.values(), "*")
    plt.show()


def titleTagTest():
    _train_path = "../../dataset/oppo_round1_train_20180929/oppo_round1_train_20180929.txt"
    valid_path = "../../dataset/oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt"
    train_records = File.readList(_train_path)
    [prefix, predict, title, tag, label] = OppDataFormat.withResult(train_records)
    max_predict = _maxPredict(predict)

    title_words = list(map(lambda x: list(filter(lambda word: word not in [""], jb.cut(x, cut_all=True))), title))
    title_tag = []
    title_tag_label = []
    related = BinaryClassifierStatistician()
    r_size = len(label)
    for idx in range(r_size):
        un = list(set(title_words[idx]))
        for w in un:
            title_tag.append(w + tag[idx])
            title_tag_label.append(label[idx])
    # 添加历史记录
    related.fit(title_tag, title_tag_label)
    print(len(related.get(title_tag)))


def prefixTagTest():
    pass

def predictTagTest():
    pass

if __name__ == "__main__":
    # 正样本比率:744195/200000=0.3720975

    # sameTest()
    # prefix和title一致时,正样本比率:163746/385094=0.42521
    # max_prefix和title一致时,正样本比率:213345/375572=0.56805
    # prefix,max_prefix,title均一致时,正样本比率:0/0

    # similarTest()


    titleTagTest()
    #
