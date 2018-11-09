class List:

    @staticmethod
    def extract(data, index: list):
        """
        index按'1取0不取'的方式从data中取出相应的数据
        :param data:
        :param index:
        :return:
        """
        res = []

        for idx in range(len(index)):
            if index[idx] in [1, '1']:
                res.append(data[idx])
        return res

    @staticmethod
    def merge(x, y, choseX):
        """
        合并x,y数组,choseX为合并x的索引
        :param x:
        :param y:
        :param choseX:
        :return:
        """
        res = []
        current_x = 0
        current_y = 0

        check = 0
        for idx in range(len(x) + len(y)):
            if check < len(choseX) and idx == choseX[check]:
                check += 1
                res.append(x[current_x])
                current_x += 1
            else:
                res.append(y[current_y])
                current_y += 1
        return res

    @staticmethod
    def distinct(x: list):
        """
        数据去重
        返回:1.新数组,2.重复数,3.该数第一次出现的索引
        :param x:
        :return:[new_x, rep, frt_index]
        """
        unique = {}

        # 结果
        cursor = 0
        new_x = []
        rep = []
        frt_index = []

        for idx in range(len(x)):
            r = x[idx]
            if isinstance(r, (list, dict, tuple)):
                key = ""
                for t in r:
                    key += str(t)
            else:
                key = r
            find = unique.get(key)
            if find is None:
                unique.setdefault(key, cursor)
                new_x.append(r)
                rep.append(1)
                frt_index.append(idx)
                cursor += 1
            else:
                rep[find] += 1

        return [new_x, rep, frt_index]
