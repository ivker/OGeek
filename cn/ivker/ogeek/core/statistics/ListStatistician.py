import logging
from functools import reduce

import numpy as np
from cn.ivker.ogeek.core.statistics.Statistician import Statistician

logger = logging.getLogger("statistician")


class ListStatistician(Statistician):

    def __init__(self):
        # 每项前N列为特征,后1项为频数
        super().__init__()

    def fit(self, x):
        """
        统计频数
        :param x:
        :return:
        """
        x = x.copy()
        # 历史样本字典
        statistics = self.statistics
        for r in x[:]:
            key = str(reduce(lambda a, b: str(a) + str(b), r))

            if statistics.get(key) is not None:
                statistics[key] += 1
            else:
                statistics.setdefault(key, 1)

        # 提示
        logger.info("statistician在fit后,规模为%d" % len(statistics))

    def get(self, x: list):
        """
        获取统计量
        :param x:
        :return: 数量
        """
        res = []
        statistics = self.statistics
        for r in x[:]:
            key = str(reduce(lambda a, b: str(a) + str(b), r))

            st = statistics.get(key)
            if st is not None:
                res.append(st)
            else:
                res.append(0)

        return res
