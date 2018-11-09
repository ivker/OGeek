import logging

import numpy as np

from cn.ivker.ogeek.util.Dict import Dict

logger = logging.getLogger("Array")


class Array:
    @staticmethod
    def extract(data, selected):
        """
        按selected从data中取出相应的数据
        :param data:
        :param selected:
        :return:
        """
        if len(selected) > 0:
            return data[selected]
        else:
            return np.asarray([])

    @staticmethod
    def merge(x, y, choseX):
        """
        合并x,y数组,choseX为合并x的索引
        :param x:
        :param y:
        :param choseX:
        :return:
        """
        if x.shape < 1:
            return y
        r_size = x.shape[0] + y.shape[0]
        c_size = x.shape[1]
        res = np.zeros((r_size, c_size))
        current_x = 0
        current_y = 0

        # 当前位置选择x/y
        check = 0
        for idx in range(r_size):
            if idx == choseX[check]:
                res[idx] = (x[current_x])
                check += 1
                current_x += 1
            else:
                res[idx] = (y[current_y])
                current_y += 1
        return res

    @staticmethod
    def crossingIndex(a, b, a_dict=None, b_dict=None):
        """
        返回a,b两个数组的交集,差集所对应的数组下标
        :param a:
        :param b:
        :param a_dict:
        :param b_dict:
        :return:
        """
        if a_dict is None:
            a_dict = Dict.array2IndexDict(a)
        if b_dict is None:
            b_dict = Dict.array2IndexDict(b)
        # int:交集,diff:差集
        a_int_index = []
        a_diff_index = []
        b_int_index = []
        b_diff_index = []
        inter = set(a_dict) & set(b_dict)

        for i in a_dict:
            if i in inter:
                a_int_index.extend(a_dict[i])
            else:
                a_diff_index.extend(a_dict[i])
        for i in b_dict:
            if i in inter:
                b_int_index.extend(b_dict[i])
            else:
                b_diff_index.extend(b_dict[i])

        logger.info("集合A(%d条记录) 和 集合B(%d条记录) 的交集为 %d 条记录" % (len(a_dict), len(b_dict), len(inter)))
        a_int_index.sort()
        b_int_index.sort()
        return [np.asarray(a_int_index), np.asarray(a_diff_index),  np.asarray(b_int_index),
                np.asarray(b_diff_index)]
