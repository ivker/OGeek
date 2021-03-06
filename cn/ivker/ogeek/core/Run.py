import logging
import time

import numpy as np

from cn.ivker.ogeek.core.encoder.oneHot import oneHot
from cn.ivker.ogeek.core.models.rf.RFClassifier import RFClassifier
from cn.ivker.ogeek.core.models.rf.RFRegressor import RFRegressor
from cn.ivker.ogeek.core.models.svm.SVM import SVM
from cn.ivker.ogeek.core.preprocess.OppClassifiedData import OppClassifiedData
from cn.ivker.ogeek.core.preprocess.OppDataFormat import OppDataFormat
from cn.ivker.ogeek.core.preprocess.OppRegressedData import OppRegressedData
from cn.ivker.ogeek.core.preprocess.Regress import Regress
from cn.ivker.ogeek.core.valider.Valid import Valid
from cn.ivker.ogeek.util.Array import Array
from cn.ivker.ogeek.util.File import File
from cn.ivker.ogeek.util.List import List

logger = logging.getLogger("Run")


def loadSVM():
    model_path = "../result/trained/svm.m"
    svm = SVM.load(model_path)
    return svm


def trainSVM(x_train, y_train, encoder=None):
    svm = SVM(encoder=encoder)
    logger.info('开始训练SVM模型')
    svm.fit(x_train, y_train)

    # *.保存模型/读取模型
    # model_name = "../result/trained/svm" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".m"
    # svm.dump(model_name)
    return svm


def trainRF(x_train, y_train, encoder=None, estimators=10):
    rf = RFClassifier(encoder=encoder, n_estimators=estimators)
    logger.info('开始训练RF模型')
    rf.fit(x_train, y_train)

    # *.保存模型/读取模型
    # model_name = "../result/trained/rf" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".m"
    # rf.dump(model_name)
    return rf


def loadRF():
    model_path = "../result/trained/rf.m"
    rf = RFRegressor().load(model_path)
    return rf


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    tmp_path = "../dataset/test.txt"
    train_path = "../dataset/oppo_round1_train_20180929/oppo_round1_train_20180929.txt"
    valid_path = "../dataset/oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt"
    cur_path = "../dataset/oppo_round1_cur/oppo_round1_cur.txt"
    test_path = "../dataset/oppo_round1_test_A_20180929/oppo_round1_test_A_20180929.txt"

    # 存储路径
    x_m1_path = "../preprocessed/train/x1/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    x_m2_path = "../preprocessed/train/x2/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    y_m1_path = "../preprocessed/train/y1/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    y_m2_path = "../preprocessed/train/y2/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    vx_m1_path = "../preprocessed/valid/x1/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    vx_m2_path = "../preprocessed/valid/x2/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    vy_m1_path = "../preprocessed/valid/y1/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    vy_m2_path = "../preprocessed/valid/y2/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    vy_index_path = "../preprocessed/valid/index/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    vy_path = "../preprocessed/valid/y/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    tx_m1_path = "../preprocessed/test/x1/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    tx_m2_path = "../preprocessed/test/x2/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    tx_index_path = "../preprocessed/test/index/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    # x_m2_dict_path = "../preprocessed/train/dict/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"

    result_valid_m1_regress_path = "../result/valid/m1/regress/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_valid_m1_classified_path = "../result/valid/m1/classified/" + str(
        time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_valid_m2_regress_path = "../result/valid/m2/regress/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_valid_m2_classified_path = "../result/valid/m2/classified/" + str(
        time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_valid_classified_path = "../result/valid/classified/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_valid_f1_path = "../result/valid/f1/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_test_m1_regress_path = "../result/test/m1/regress/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_test_m1_classified_path = "../result/test/m1/classified/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_test_m2_regress_path = "../result/test/m2/regress/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_test_m2_classified_path = "../result/test/m2/classified/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"
    result_test_classified_path = "../result/test/classified/" + str(time.strftime("%Y-%m-%d %H_%M_%S")) + ".txt"

    # 读取数据路径
    load_x_m1_path = "../preprocessed/train/x1/x.txt"
    load_x_m2_path = "../preprocessed/train/x2/x.txt"
    load_y_m1_path = "../preprocessed/train/y1/y.txt"
    load_y_m2_path = "../preprocessed/train/y2/y.txt"
    load_vx_m1_path = "../preprocessed/valid/x1/x.txt"
    load_vx_m2_path = "../preprocessed/valid/x2/x.txt"
    load_vy_m1_path = "../preprocessed/valid/y1/y.txt"
    load_vy_m2_path = "../preprocessed/valid/y2/y.txt"
    load_vx_index_path = "../preprocessed/valid/index/i.txt"
    load_vy_path = "../preprocessed/valid/y/y.txt"
    load_tx_m1_path = "../preprocessed/test/x1/x.txt"
    load_tx_m2_path = "../preprocessed/test/x2/x.txt"
    load_tx_index_path = "../preprocessed/test/index/i.txt"

    load_m1_regress_path = "../result/valid/m1/regress/y.txt"
    load_m1_classified_path = "../result/valid/m1/classified/y.txt"
    load_m2_regress_path = "../result/valid/m2/regress/y.txt"
    load_m2_classified_path = "../result/valid/m2/classified/y.txt"

    # 配置
    _train_path = cur_path
    _valid_path = None
    _test_path = test_path

    # 生成/读取数据
    gen_data = False

    run_model1 = True
    run_model2 = True
    load_model1 = False
    load_model2 = False

    # *.预处理模型
    opC = OppClassifiedData()
    opR = OppRegressedData()

    # 特征处理后的中文编码维度 编码器
    m1_chinese_indexes = [0, 1, 2]
    m2_chinese_indexes = [0]

    m1_f1_threshold = 0.36
    m2_f1_threshold = 0.48
    m1_n_estimators = 30
    m2_n_estimators = 30

    # ###########启动############
    # 时间戳
    logger.info("Start at " + time.strftime("%Y-%m-%d %H_%M_%S"))
    if gen_data:
        # *.执行数据预处理/读取已处理数据
        # *.从样本数据中获取特征
        # 区块内存
        if True:
            train_records = File.readList(_train_path)
            [prefix, predict, title, tag, label] = OppDataFormat.withResult(train_records)
            [x, y] = opC.fitTrain(prefix, predict, title, tag, label)
            logger.info("已fit Model1")
            opR.fitTrain(prefix, predict, title, tag, label)
            logger.info("已fit Model2")
        # *.从验证数据中获取特征,结果
        if _valid_path is not None:
            valid_records = File.readList(_valid_path)
            [prefix, predict, title, tag, label] = OppDataFormat.withResult(valid_records)
            [vx, vy] = opC.fitValid(prefix, predict, title, tag, label)
            opR.fitValid(prefix, predict, title, tag, label)
        else:
            vx = []
            vy = []

        # *.从测试数据中获取特征
        if _test_path is not None:
            test_records = File.readList(_test_path)
            [prefix, predict, title, tag] = OppDataFormat.featureOnly(test_records)
            tx = opC.fitTest(prefix, predict, title, tag)
            opR.fitTest(prefix, predict, title, tag)
        else:
            tx = []

        # 对训练数据 按模型1(分类)和 模型2(回归)进行处理
        # model 1 的训练数据:取与预测相关的样本,做分类模型
        # *.测相关的样本:合并valid和test样本

        if _valid_path is not None:
            [x_int_index, x_diff_index, v_int_index, v_diff_index] = Array.crossingIndex(x, vx)
            x_m1_index = x_int_index
            x_m2_index = x_diff_index
            vx_m1_index = v_int_index
            vx_m2_index = v_diff_index
            [x_train_m1, y_train_m1] = opC.getTrain(x_m1_index)
            [x_valid_m1, y_valid_m1] = opC.getValid(vx_m1_index)
            [x_train_m2, y_train_m2] = opR.getTrain(x_m2_index)
            [x_valid_m2, y_valid_m2] = opR.getValid(vx_m2_index)

            File.writeListInList(vx_m1_path, x_valid_m1)
            File.writeList(vy_m1_path, y_valid_m1)
            File.writeListInList(vx_m2_path, x_valid_m2)
            File.writeList(vy_m2_path, y_valid_m2)
            File.writeList(vy_index_path, vx_m1_index)
            File.writeList(vy_path, vy)
        else:
            [x_int_index, x_diff_index, t_int_index, t_diff_index] = Array.crossingIndex(x, tx)
            x_m1_index = x_int_index
            x_m2_index = x_diff_index
            t_m1_index = t_int_index
            t_m2_index = t_diff_index
            [x_train_m1, y_train_m1] = opC.getTrain(x_m1_index)
            x_test_m1 = opC.getTest(t_m1_index)
            [x_train_m2, y_train_m2] = opR.getTrain(x_m2_index)
            x_test_m2 = opR.getTest(t_m2_index)
            x_test_index = t_m1_index

            File.writeListInList(tx_m1_path, x_test_m1)
            File.writeListInList(tx_m2_path, x_test_m2)
            File.writeList(tx_index_path, t_m1_index)

        # 保存train
        if True:
            File.writeListInList(x_m1_path, x_train_m1)
            File.writeList(y_m1_path, y_train_m1)
            File.writeListInList(x_m2_path, x_train_m2)
            File.writeList(y_m2_path, y_train_m2)

    # 读取数据
    else:
        if run_model1:
            x_train_m1 = File.read2DArray(load_x_m1_path)
            y_train_m1 = File.read1DArray(load_y_m1_path)
            x_valid_m1 = File.read2DArray(load_vx_m1_path)
            y_valid_m1 = File.read1DArray(load_vy_m1_path)
            x_test_m1 = File.read2DArray(load_tx_m1_path)
        else:
            x_train_m1 = np.asarray([])
            y_train_m1 = np.asarray([])
            x_valid_m1 = np.asarray([])
            y_valid_m1 = np.asarray([])
            x_test_m1 = np.asarray([])

        if run_model2:
            x_train_m2 = File.read2DArray(load_x_m2_path)
            y_train_m2 = File.read1DArray(load_y_m2_path)
            x_valid_m2 = File.read2DArray(load_vx_m2_path)
            y_valid_m2 = File.read1DArray(load_vy_m2_path)
            x_test_m2 = File.read2DArray(load_tx_m2_path)
        else:
            x_train_m2 = np.asarray([])
            y_train_m2 = np.asarray([])
            x_valid_m2 = np.asarray([])
            y_valid_m2 = np.asarray([])
            x_test_m2 = np.asarray([])

        if _valid_path is not None:
            y_valid_m1 = File.read1DArray(load_vy_m1_path)
            y_valid_m2 = File.read1DArray(load_vy_m2_path)
            vx_m1_index = File.read1DArray(load_vx_index_path, type=np.int32)
            vy = File.read1DArray(load_vy_path)
        else:
            vx_m1_index = np.asarray([])
            vy = np.asarray([])

        if _test_path is not None:
            tx_m1 = File.read2DArray(load_tx_m1_path)
            tx_m2 = File.read2DArray(load_tx_m2_path)
            tx_index = File.read1DArray(load_tx_index_path, type=np.int32)

    # *.查看样本分布
    # x_train_dict = Dict.array2IndexDict(x)
    # File.writeDict(train_dict_path, x_train_dict)

    # 执行验证用例

    if _valid_path is not None:
        # 运行模型1
        if run_model1:
            # *.运行rf模型
            m = trainRF(x_train_m1, y_train_m1, encoder=oneHot(m1_chinese_indexes), estimators=m1_n_estimators)
            # m1 = loadRF()

            # *.输出预测值
            regress1 = m.predict(x_valid_m1)
            # *.回归数据变为分类数据
            classified1 = Regress.result2BinaryClassified(regress1, m1_f1_threshold)
            if result_valid_m1_regress_path is not None:
                File.writeList(result_valid_m1_regress_path, regress1)
            if result_valid_m1_classified_path is not None:
                File.writeList(result_valid_m1_classified_path, classified1)
        if load_model1:
            regress1 = File.read1DArray(load_m1_regress_path)
            classified1 = File.read1DArray(load_m1_classified_path)

        # 运行模型2
        if run_model2:
            m2 = trainRF(x_train_m2[0:40000], y_train_m2[0:40000], encoder=oneHot(m2_chinese_indexes),
                         estimators=m2_n_estimators)
            regress2 = m2.predict(x_valid_m2)
            # *.回归数据变为分类数据
            classified2 = Regress.result2BinaryClassified(regress2, m2_f1_threshold)
            if result_valid_m2_regress_path is not None:
                File.writeList(result_valid_m2_regress_path, regress2)
            if result_valid_m2_classified_path is not None:
                File.writeList(result_valid_m2_classified_path, classified2)
        if load_model2:
            regress2 = File.read1DArray(load_m2_regress_path)
            classified2 = File.read1DArray(load_m2_classified_path)

        # *.评估分模型预测结果
        if run_model1 or load_model1:
            logger.info("评估m1预测结果:")
            Valid.f1(classified1, y_valid_m1)
        if run_model2 or load_model2:
            logger.info("评估m2预测结果:")
            Valid.f1(classified2, y_valid_m2)

        # 评估整体预测结果
        if (run_model1 or load_model1) and (run_model2 or load_model2):
            logger.info("综合评估预测结果:")
            classified = List.merge(classified1, classified2, vx_m1_index)
            Valid.f1(classified, vy, result_valid_f1_path)
            File.writeList(result_valid_classified_path, classified)

    # 执行测试用例
    if _test_path is not None:
        # 运行模型1
        if run_model1:
            # *.运行rf模型
            m = trainRF(x_train_m1, y_train_m1, encoder=oneHot(m1_chinese_indexes), estimators=m1_n_estimators)
            # m1 = loadRF()

            # *.输出预测值
            regress1 = m.predict(x_test_m1)
            # *.回归数据变为分类数据
            classified1 = Regress.result2BinaryClassified(regress1, m1_f1_threshold)
            if result_test_m1_regress_path is not None:
                File.writeList(result_test_m1_regress_path, regress1)
            if result_test_m1_classified_path is not None:
                File.writeList(result_test_m1_classified_path, classified1)
        if load_model1:
            regress1 = File.read1DArray(load_m1_regress_path)
            classified1 = File.read1DArray(load_m1_classified_path)

        # 运行模型2
        if run_model2:
            m2 = trainRF(x_train_m2, y_train_m2, encoder=oneHot(m2_chinese_indexes), estimators=m2_n_estimators)
            regress2 = m2.predict(x_test_m2)
            # *.回归数据变为分类数据
            classified2 = Regress.result2BinaryClassified(regress2, m2_f1_threshold)
            if result_test_m2_regress_path is not None:
                File.writeList(result_test_m2_regress_path, regress2)
            if result_test_m2_classified_path is not None:
                File.writeList(result_test_m2_classified_path, classified2)
        if load_model2:
            regress2 = File.read1DArray(load_m1_regress_path)
            classified2 = File.read1DArray(load_m1_classified_path)

        classified = List.merge(classified1, classified2, x_test_index)
        File.writeList(result_test_classified_path, classified)

    # 时间戳
    logger.info("End at " + time.strftime("%Y-%m-%d %H_%M_%S"))
