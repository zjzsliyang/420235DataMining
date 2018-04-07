import math
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot


def kmeans(df, random_vip, knns):
    k = int(math.sqrt(df.shape[1]) / 2)
    silhouette_avgs = []

    for n_clusters in range(2, k + 2):
        logging.debug("KMeans: n_clusters = {}".format(n_clusters))
        clusterer = KMeans(n_clusters=n_clusters)
        X = StandardScaler().fit_transform(df.T)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avgs.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score in KMeans is :", silhouette_avg)

        res = 0
        no = cluster_labels[df.columns.get_loc(random_vip)]
        for neighbor in knns:
            if cluster_labels[df.columns.get_loc(neighbor)] == no:
                res += 1
            else:
                logging.info(
                    "KMeans: vipno: {} is not in the same cluster.".format(
                        neighbor))
        print(
            "For k = {} in kNN, there has {} in the same cluster in KMeans.".format(
                len(knns), res))

    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.xlim([2, k + 1])
    matplotlib.pyplot.ylim([-0.1, 1])
    matplotlib.pyplot.xlabel(
        "The number of clusters as well as the number of centroids")
    matplotlib.pyplot.ylabel("The silhouette coefficient values")
    matplotlib.pyplot.title(
        "Silhouette analysis for KMeans clustering on reco data",
        fontsize=14, fontweight='bold', y=1.08)
    matplotlib.pyplot.plot(range(2, k + 2), silhouette_avgs)
    matplotlib.pyplot.savefig('res/kmeans.png')

    return silhouette_avgs.index(max(silhouette_avgs)) + 2
