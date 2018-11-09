# 统计:相对于历史记录中,预测记录中:
# *.学习/预测样本中重复样本比例
# *.预测/样本在学习样本中的重复比例
# *.预测样本各维度的新特征比例
# *.预测样本各维度致使新样本的比例
from cn.ivker.ogeek.core.preprocess.GenFeatures import GenFeatures
from cn.ivker.ogeek.core.preprocess.GenResult import GenResult


def duplicate_self(old_x):
    # 特征映射字典
    feature_mapping = {}

    # 结果
    new_x = []

    # 将样本按特征进行映射,保存相同特征的样本下标
    for idx in range(len(old_x)):
        key = ''
        for feature in old_x[idx]:
            key += str(feature)
        if feature_mapping.get(key) is None:
            feature_mapping.setdefault(key, [idx])
        else:
            feature_mapping[key].append(idx)

    # 遍历相同特征的记录
    for feature in feature_mapping:
        # 保存特征及计数
        idx_list = feature_mapping[feature]
        current_x = old_x[idx_list[0]]

        new_x.append(current_x)

    # 提示
    print('原记录数:%d,去重后记录数:%d' % (len(old_x), len(new_x)))


def duplicate_predict2train(x_train, x_predict):
    # 特征映射字典
    feature_mapping = {}

    # 结果
    exist_predict = []

    # 将样本按特征进行映射,保存相同特征的样本下标
    for record in x_train:
        key = ''
        for feature in record:
            key += str(feature)
        if feature_mapping.get(key) is None:
            feature_mapping.setdefault(key, "")

    # 遍历相同特征的记录
    for record in x_predict:
        # 保存特征及计数
        key = ''
        for feature in record:
            key += str(feature)
        if feature_mapping.get(key) is not None:
            exist_predict.append(record)

    # 提示
    print('原记录数:%d,存在学习样本中的记录数:%d' % (len(x_predict), len(exist_predict)))


def duplicate_feature(x_train, x_predict):
    # 特征映射字典:每个特征分开
    feature_mapping_list = []
    feature_exist_list=[]
    feature_miss_list = []
    feature_size = len(x_train[0])
    for idx in range(feature_size):
        feature_mapping_list.append({})
        feature_exist_list.append({})
        feature_miss_list.append({})

    # 将样本按特征进行映射,保存相同特征的样本下标
    for record in x_train:
        for idx in range(feature_size):
            if feature_mapping_list[idx].get(record[idx]) is None:
                feature_mapping_list[idx].setdefault(record[idx], "")

    for record in x_predict:
        for idx in range(feature_size):
            if feature_mapping_list[idx].get(record[idx]) is not None:
                feature_exist_list[idx].setdefault(record[idx], "")
            else:
                feature_miss_list[idx].setdefault(record[idx], "")
    # 提示
    for idx in range(feature_size):
        print("维度%d,已存在值数量:%d,未识别值数量:%d"
              % (idx, len(feature_exist_list[idx].keys()), len(feature_miss_list[idx].keys())))


tmp_path = "../../dataset/test.txt"
train_path = "../../dataset/oppo_round1_train_20180929/oppo_round1_train_20180929.txt"
valid_path = "../../dataset/oppo_round1_vali_20180929/oppo_round1_vali_20180929.txt"
cur_path = "../../dataset/oppo_round1_cur/oppo_round1_cur.txt"
test_path = "../../dataset/oppo_round1_test_A_20180929/oppo_round1_test_A_20180929.txt"

# 1.学习-从学习数据中获取特征
x_train_feature = GenFeatures.get(train_path)

# *.学习,特征的监督结果
y_train_feature = GenResult.get(train_path)

# *.验证-从验证数据中获取特征
x_predict_feature = GenFeatures.get(test_path)

# *.统计学习/预测样本的完全特征重复率
# print("对学习样本去重:")
# duplicate_self(x_train_feature)
# print("对预测样本去重:")
# duplicate_self(x_predict_feature)
# print("预测样本在学习样本中的重复数:")
# duplicate_predict2train(x_train_feature, x_predict_feature)
print("预测样本各维度的新特征比例:")
duplicate_feature(x_train_feature, x_predict_feature)
