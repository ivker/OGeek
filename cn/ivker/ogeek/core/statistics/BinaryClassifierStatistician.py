import logging

from cn.ivker.ogeek.core.statistics.Statistician import Statistician

logger = logging.getLogger("Classifier")


class BinaryClassifierStatistician(Statistician):

    def __init__(self):
        # 每项前N列为特征,后2项为标签1/0的频数
        super().__init__()
        self.statistics = {}

    def fit(self, x: list, y: list):
        """
        统计频数:
        :param x:每个对象仍可以是list
        :param y:二分结果
        :return:["_特征值":[1数量,0数量]]
        """
        x = x.copy()
        # 历史样本字典
        statistics = self.statistics

        for idx in range(len(x)):
            r = x[idx]
            if isinstance(r, (list, dict, tuple)):
                key = ""
                for f in r:
                    key += str(f)
            else:
                key = r

            if statistics.get(key) is not None:
                if y[idx] in [1, '1']:
                    statistics[key][0] += 1
                else:
                    statistics[key][1] += 1
            else:
                if y[idx] in [1, '1']:
                    statistics.setdefault(key, [1, 0])
                else:
                    statistics.setdefault(key, [0, 1])

        logger.info("statistics,fit后的规模为%d" % len(statistics))

    def get(self, x: list):
        """
        获取统计量
        :param x:
        :return: ["正样本(1)数量"]，["负样本(1)数量"]
        """
        p = []
        n = []
        statistics = self.statistics
        for r in x:
            if isinstance(r, (list, dict, tuple)):
                key = ""
                for f in r:
                    key += str(f)
            else:
                key = r

            st = statistics.get(key)
            if st is not None:
                p.append(st[0])
                n.append(st[1])
            else:
                p.append(0)
                n.append(0)

        return [p, n]
