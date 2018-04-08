import numpy
import logging
import matplotlib.pyplot
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def dbscan(df, random_vip, knns):
    silhouette_avgs = []
    for eps in numpy.arange(0.1, 1, 0.1):
        logging.debug("DBSCAN: eps = {}".format(eps))
        X = StandardScaler().fit_transform(df.T)
        db = DBSCAN(eps=eps).fit(X)
        core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        cluster_labels = db.labels_
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        silhouette_avg = silhouette_score(X, db.labels_)
        silhouette_avgs.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score in DBSCAN is :", silhouette_avg)

        res = 0
        no = cluster_labels[df.columns.get_loc(random_vip)]
        for neighbor in knns:
            if cluster_labels[df.columns.get_loc(neighbor)] == no:
                res += 1
            else:
                logging.info("DBSCAN: vipno: {} is not in the same cluster.".format(
                    neighbor))
        print("For k = {} in kNN, there has {} in the same cluster in DBSCAN.".format(
            len(knns), res))

    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.xlim(numpy.arange(0.1, 1, 0.1))
    matplotlib.pyplot.ylim([-0.1, 1])
    matplotlib.pyplot.xlabel("The number of clusters as well as the number of centroids")
    matplotlib.pyplot.ylabel("The silhouette coefficient values")
    matplotlib.pyplot.suptitle(
        "Silhouette analysis for DBSCAN clustering on reco data",
        fontsize=14, fontweight='bold')
    matplotlib.pyplot.plot(silhouette_avgs)
    matplotlib.pyplot.show()

    return silhouette_avgs.index(max(silhouette_avgs)) / 10
