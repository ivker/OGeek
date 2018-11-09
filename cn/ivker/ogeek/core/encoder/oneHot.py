import time

from cn.ivker.ogeek.util.File import File

import logging

logger = logging.getLogger("oneHot")


class oneHot:
    """
        将数据中文字进行编码
        每一个编码器的每一个维度拥有独立的字典
    """

    def __init__(self, features: list):
        """

        :param features:编码维度
        """
        # 记录编码维度
        self.features = features
        # 编码字典:codes(维度)(原值)=编码值
        self.codes = {}
        for feature in features:
            self.codes[feature] = {}

    def oneHot(self, records: list, new='add'):
        """
        将中文数据进行编码
        :param records:数据,每一条记录含若干维度的特征
        :param new:若数据为新增时处理:add表示新增,none表示置空(-1)
        :return:
        """

        new_words = {}

        for record in records:
            for feature in self.features:
                coder = self.codes[feature]
                value = record[feature]
                code = coder.get(value, -1)
                # 是否在不存在时新增
                if code == -1:
                    if new == 'add':
                        code = len(coder)
                        coder.setdefault(value, code)
                    else:
                        new_words.setdefault(value, "-1")

                record[feature] = code

        # 新数据提示并保存
        if len(new_words) > 0:
            logger.info("Encoder found %d new word(s)" % len(new_words))
            result_regress_path = "../result/newwords/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
            # File.writeList(result_regress_path, list(new_words.keys()))

        return records
