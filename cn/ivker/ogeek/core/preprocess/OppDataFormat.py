class OppDataFormat:

    @staticmethod
    def withResult(records: list):
        """
        将数据按"oppo"数据格式化
        格式:
        前缀  {"预测词":"概率"}    title  tag label
        :param records:
        :return:
        [[前缀,title,tag],[{"预测词":"概率"}],[label]]
        """

        # 结果
        prefix = []
        title = []
        tag = []
        predict = []
        label = []

        for record in records:
            # 切割出原始数据
            item = record.rstrip('\n').split('\t')
            if len(item) < 1:
                continue
            # 处理预测结果并保存(可能为空,最多10项)
            pres = item[1].strip('{').strip('}').replace(" ", "").split(',')
            predicts = {}
            for pre in pres:
                pv = pre.split('":"')
                key = pv[0].replace('"', '')
                if key != "" and key is not None and len(pv) > 1:
                    predicts[key] = float(pv[1].replace('"', ''))

            # 存储
            prefix.append(item[0])
            title.append(item[2])
            tag.append(item[3])
            predict.append(predicts)
            if len(item) > 4:
                label.append(item[4])

        return [prefix, predict, title, tag, label]

    @staticmethod
    def featureOnly(records: list):
        [prefix, predict, title, tag, label] = OppDataFormat.withResult(records)
        return [prefix, predict, title, tag]
