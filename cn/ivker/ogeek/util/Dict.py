import logging

logger = logging.getLogger("Dict")


class Dict:

    @staticmethod
    def array2IndexDict(a):
        """
        将array转为索引字典:
        字典的Key为list属性值,Value为出现该属性值的索引
        :param a:
        :return:
        """
        if a.shape[0] < 1:
            return {}

        r_size = a.shape[0]
        c_size = a.shape[1]
        d = {}

        for r in range(r_size):
            key = ''
            for c in range(c_size):
                key += str(a[r][c])

            ids = d.get(key)
            if ids is None:
                d.setdefault(key, [r])
            else:
                ids.append(r)

        logger.info("记录数:%d,被压缩为%d的字典" % (r_size, len(d)))
        return d

    @staticmethod
    def list2IndexDict(l: list):
        """
        将list转为索引字典:
        字典的Key为list属性值,Value为出现该属性值的索引
        :param l:
        :return:
        """
        size = len(l)
        d = {}

        for idx in range(size):
            key = ''
            features = l[idx]
            if isinstance(features, (list, tuple, dict)):
                for feature in features:
                    key += str(feature)
            else:
                key = features

            ids = d.get(key)
            if ids is None:
                d.setdefault(key, [idx])
            else:
                ids.append(idx)

        return d

    @staticmethod
    def extract(d: dict, keys):
        """
        返回包含keys的局部字典
        :param d:
        :param keys:
        :return:
        """
        n = {}
        for key in keys:
            n.setdefault(key, d.get(key))

        return n
