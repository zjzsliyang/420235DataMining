import csv
import math
import pandas
import logging
import matplotlib.pyplot as plt
from lshash.lshash import LSHash
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

''' raw reco data info
db_name   说明        index        example
uid     订单编号        0        160203104111518000
sldat   购买时间        1        2016-08-12 09:42:46
pno     收银员编号      2        15
cno     收银机编号      3        8331
vipno   会员编号        4        2900000161443
id      商品单内编号     5        3
pluno   商品编号        6        14721041
bcd     条码            7        6903252713411
pluname 商品名称        8        康师傅面霸煮面上汤排骨面五入100g*5
spec    包装规格        9        1*6
pkunit  商品单位        10       包
dptno   商品类型编号    11        14721
dptname 商品类型名称    12        连包装
bndno   品牌编号       13        14177
bndname 品牌名称       14        康师傅方便面
qty     购买数量       15        1
amt     金额           16        17.5
disamt  是否打折       17        0  
ismmx   是否促销       18        0
mtype   促销类型       19        0
mdocno  促销单号       20
'''


def read_data():
    trade = {}
    with open('../data/reco_data/trade.csv', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            vipno = row[4]
            pluno = row[6]
            amt = float(row[16])
            if vipno not in trade:
                trade[vipno] = {}
            trade[vipno][pluno] = amt + trade[vipno].get(pluno, 0)
    # dataframe structure:
    #          vipno
    # pluno    amt
    df = pandas.DataFrame(trade).fillna(0).round(0)
    logging.debug("dataframe shape: {}".format(df.shape))
    return df


def knn(df, k, coefficient):
    hash_size = int(coefficient) * df.shape[1]
    lsh = LSHash(hash_size, input_dim=df.shape[0])
    for vipno in df:
        lsh.index(df[vipno], extra_data=vipno)
    random_column = df[df.columns.to_series().sample(1)]
    random_vip = random_column.columns.values[0]
    logging.info("random vipno: {}".format(random_vip))
    res = lsh.query(random_column.values.flatten())[1:k + 1]
    print("vipno in ranked order using kNN(k={}):".format(k))
    knns = []
    for item in res:
        print(item[0][1])
        knns.append(item[0][1])
    return random_vip, knns


def kmeans(df, random_vip, knns):
    k = int(math.sqrt(df.shape[1]) / 2)
    silhouette_avgs = []

    for n_clusters in range(k, k + 1):
        logging.debug("n_clusters = {}".format(n_clusters))
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(df.T)
        silhouette_avg = silhouette_score(df.T, cluster_labels)
        silhouette_avgs.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        res = 0
        no = cluster_labels[df.columns.get_loc(random_vip)]
        for neighbor in knns:
            if cluster_labels[df.columns.get_loc(neighbor)] == no:
                res += 1
        print("For k = {} in kNN, there has {} in the same cluster.".format(len(knns), res))

    plt.figure(figsize=(10, 6))
    plt.xlim([2, k + 1])
    plt.ylim([-0.1, 1])
    plt.xlabel("The number of clusters as well as the number of centroids")
    plt.ylabel("The silhouette coefficient values")
    plt.suptitle(
        "Silhouette analysis for KMeans clustering on reco data",
        fontsize=14, fontweight='bold')
    plt.plot(range(2, k + 2), silhouette_avgs)
    plt.show()


def main():
    logging.basicConfig(level=logging.DEBUG)

    df = read_data()

    # Problem I
    coefficients = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    ks = [1, 2, 3, 4, 5]
    random_vip, knns = knn(df, ks[4], coefficients[4])

    # Problem II
    kmeans(df, random_vip, knns)

    # Problem III

    # Problem IV


if __name__ == '__main__':
    main()
