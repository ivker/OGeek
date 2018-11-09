from cn.ivker.ogeek.core.encoder.Encoder import Encoder
from cn.ivker.ogeek.core.preprocess.GenFeatures import GenFeatures
from cn.ivker.ogeek.core.preprocess.GenResult import GenResult
from cn.ivker.ogeek.core.preprocess.Regress import Regress
from cn.ivker.ogeek.core.statistics.Statistics import Statistics

tmp_path = "../../dataset/test.txt"
train_path = "../../dataset/oppo_round1_train_20180929/oppo_round1_train_20180929.txt"

# 特征处理后的中文编码维度 编码器
chinese_indexes = [0, 2, 3]
encoder = Encoder(chinese_indexes)
stat = Statistics()

# 1.学习-从学习数据中获取特征
train_x = GenFeatures.get(tmp_path)

# # 2.学习-将特征数据进行编码
# train_x = encoder.oneHot(train_records, new='add')

# 3.学习,特征的监督结果
train_y = GenResult.get(tmp_path)

# 4.统计正负样本数
# countAll = int(len(train_y))
# count1 = len(list(filter(lambda y: y == '1', train_y)))
# count0 = countAll - count1
# print("正/负样本数:%d / %d = %.2f%% :%.2f%%" % (count1, count0, count1 / countAll * 100, count0 / countAll * 100))

# 5.特征质量 - 冲突

# 生成统计维度
train_x = train_x
train_y = train_y

[train_x, train_y] = Regress.classified2Regress(train_x, train_y)
countAll = len(train_x)
collision = 0
for i in range(countAll - 1):
    if i % (countAll / 100) == 0:
        print("已处理%.1f%%" % (i / countAll * 100))
    for j in range(i + 1, countAll):
        conflict = True
        for f in range(4):
            if train_x[i][f] != train_x[j][f]:
                conflict = False
                break
        if conflict:
            if train_y[i] != train_y[j]:
                collision += 1
                print("冲突记录:%d-%d" % (i, j))
                break
print("样本特征冲突数:%d / %d" % (collision, countAll))
