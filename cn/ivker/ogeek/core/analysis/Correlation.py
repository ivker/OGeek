import jieba as jb
import jieba.posseg as psg

# input = "葡"
predict = "葡萄酒的制作方法"
# default_predict = input
# prob = 0.077
# default_prob = 0
title = "葡萄干的功效与作用"
app = "经验"

# print([(x.word, x.flag) for x in psg.cut(input)])
# print([(x.word, x.flag) for x in psg.cut(predict)])
# print([(x.word, x.flag) for x in psg.cut(title)])
# a 形容词,b 区别词,c 连词,d 副词...

# print(list(jb.cut_for_search(predict)))# *.查询输入分词量
# (*.input,query(预测查询),title,app->分词->(去除停用词)->(近义词))
# (*.关联量=input∩title,query∩title,title∩app)
# (*.关联率=关联量/总量)
# 输入:
# *.input的字符串长度
# *.input的预测查询总概率
# *.input&title的相似度
# *.query&title的相似度
# *.title∩app的相似度
# *.title分词->各词的总点击量
# *.title分词->各词的总不点击量
# *.title的字符串长度
# *.app_id->独立维度编码表示使用的app
# *.其他app对该app的竞争成功率
# 输出:
# *.点击率
count = 0
for pr in list(jb.cut_for_search(predict)):
    if pr in list(jb.cut_for_search(title)):
        count += 1
        print(pr)
print(count)
