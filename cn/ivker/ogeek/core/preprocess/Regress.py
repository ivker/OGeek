import logging

from cn.ivker.ogeek.util.Dict import Dict

logger = logging.getLogger("Regress")


class Regress:

    @staticmethod
    def binaryClassified2Regress(old_x, old_y, old_x_dict=None):
        """
        将二分类数据转化为回归数据
        :param old_x:
        :param old_y:
        :param old_x_dict:
        :return:[new_x,new_y]
        """
        # 结果
        new_x = []
        new_y = []

        # 特征映射列表 key:特征哈希,value=记录下标
        if old_x_dict is None:
            if isinstance(old_x[0], list):
                old_x_dict = Dict.list2IndexDict(old_x)
            else:
                old_x_dict = Dict.array2IndexDict(old_x)

        # 遍历相同特征的记录
        for feature in old_x_dict:

            # 保存特征及计数
            idx_list = old_x_dict[feature]
            current_x = old_x[idx_list[0]].copy()
            current_y = old_y[idx_list[0]]
            # 正(1)/负(0)样本计数器
            positive = 1 if current_y in ['1', 1] else 0
            negative = 1 if current_y in ['0', 0] else 0

            for other in idx_list[1:]:
                positive += 1 if old_y[other] in ['1', 1] else 0
                negative += 1 if old_y[other] in ['0', 0] else 0

            new_x.append(current_x)
            new_y.append(positive / (positive + negative))

        # 提示
        logger.info('原记录数:%d,toRegress后记录数:%d' % (len(old_x), len(new_x)))

        return [new_x, new_y]

    @staticmethod
    def result2BinaryClassified(y, threshold):
        return list(map(lambda i: 1 if float(i) > threshold else 0, y))
