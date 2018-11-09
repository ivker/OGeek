import logging
import numpy

from numpy.ma import arange

from cn.ivker.ogeek.core.preprocess.Regress import Regress
from cn.ivker.ogeek.core.valider.Valid import Valid
from cn.ivker.ogeek.util.File import File
from cn.ivker.ogeek.util.List import List

logger = logging.getLogger("Tuner")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tuner_model1 = False

valid_y_path = "../preprocessed/valid/y/y.txt"
index_path = "../preprocessed/valid/index/i.txt"

m1_predict_y_regress = "../result/valid/m1/regress/y.txt"
result_path = "../result/valid/f1/f1.txt"

vy = File.readList(valid_y_path)
predicted1 = File.readList(m1_predict_y_regress)
index = File.read1DArray(index_path, type=numpy.int)

vy = list(map(lambda i: 1 if float(i) > 0.5 else 0, vy))

# *.回归数据变为分类数据
max_f1 = 0
max_m1_th = 0
for th1 in arange(0.3, 0.6, 0.01):
    classified1 = Regress.result2BinaryClassified(predicted1, th1)
    # 评估预测结果
    logger.info("threshold:%f" % th1)
    [precision, recall, accuracy, f1] = Valid.f1(classified1, vy, display=True)

    if f1 > max_f1:
        max_f1 = f1
        max_m1_th = th1

logger.info("max_f1:%f in threshold :%f" % (max_f1, max_m1_th))
