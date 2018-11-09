from cn.ivker.ogeek.util.File import File

import logging

logger = logging.getLogger("Valid")


class Valid:
    @staticmethod
    def f1(predict, result, result_path=None, display=True):
        """
        验证结果
        :param predict: 预测结果
        :param result: 真实结果
        :param result_path: 存储路径
        :param display:
        :return: [precision,recall,f1]
        """

        # 数据规模
        result_size = len(result)

        # 预测正确的正/负样本数
        true_positive_count = 0
        true_negative_count = 0

        # 预测错误的正/负样本数
        false_positive_count = 0
        false_negative_count = 0

        for idx in range(result_size):
            if result[idx] in [1, '1']:
                if predict[idx] in [1, '1']:
                    true_positive_count += 1
                else:
                    false_positive_count += 1
            else:
                if predict[idx] in [0, '0']:
                    true_negative_count += 1
                else:
                    false_negative_count += 1

        # 评估结果
        precision = -1
        recall = -1
        accuracy = -1
        f1 = -1

        if (true_positive_count + false_positive_count) != 0:
            precision = true_positive_count / (true_positive_count + false_positive_count)

        if (true_positive_count + false_negative_count) != 0:
            recall = true_positive_count / (true_positive_count + false_negative_count)

        if result_size != 0:
            accuracy = (true_positive_count + true_negative_count) / result_size

        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)

        # 输出
        con = "Precision(tp/(tp+fp)) = " + str(true_positive_count) + " / (" + str(true_positive_count) + "+" + str(
            false_positive_count) + ") = " + str(precision) + "\n"
        con += "Recall(tp/(tp+fn)) = " + str(true_positive_count) + " / (" + (
                str(true_positive_count) + " + " + str(false_negative_count)) + ") = " + str(recall) + "\n"
        con += "Accuracy((tp+tn)/all) = (" + (
                str(true_positive_count) + " + " + str(true_negative_count)) + ") / " + str(
            result_size) + " = " + str(accuracy) + "\n"
        con += "F1_Score(2*(P*R)/(P+R)) = " + str(f1)

        if result_path is not None:
            File.writelines(result_path, con)

        if display:
            logger.info("\n" + con)

        return [precision, recall, accuracy, f1]
