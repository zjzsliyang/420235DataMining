import numpy
import logging
import matplotlib.pyplot
from sklearn.cluster import DBSCAN
import palettable.colorbrewer.qualitative
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# @profile
def dbscan(df, random_vip, knns):
    silhouette_avgs = []
    hits = []
    x_min = 10
    x_max = 140
    for eps in numpy.arange(x_min, x_max, 10):
        logging.info("DBSCAN: eps = {}".format(eps))
        X = StandardScaler().fit_transform(df.T)
        db = DBSCAN(eps=eps).fit(X)
        core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        cluster_labels = db.labels_
        n_clusters = len(set(cluster_labels)) - (
            1 if -1 in cluster_labels else 0)
        silhouette_avg = silhouette_score(X, db.labels_)
        silhouette_avgs.append(silhouette_avg)
        logging.info("For n_clusters =", n_clusters,
                     "The average silhouette_score in DBSCAN is :",
                     silhouette_avg)

        hit = 0
        no = cluster_labels[df.columns.get_loc(random_vip)]
        for neighbor in knns:
            if cluster_labels[df.columns.get_loc(neighbor)] == no:
                hit += 1
            else:
                logging.debug(
                    "DBSCAN: vipno: {} is not in the same cluster.".format(
                        neighbor))
        logging.info(
            "For k = {} in kNN, there has {} in the same cluster in DBSCAN.".format(
                len(knns), hit))
        hits.append(hit)

    # plot_silhouette(x_min, x_max, silhouette_avgs)

    return silhouette_avgs.index(max(silhouette_avgs)) / 10, hits


def plot_silhouette(x_min, x_max, silhouette_avgs):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.xlim([x_min, x_max])
    matplotlib.pyplot.ylim([-0.1, 0.8])
    matplotlib.pyplot.xlabel(
        "eps (epsilon): the radius of neighborhood around a point x")
    matplotlib.pyplot.ylabel("The silhouette coefficient values")
    matplotlib.pyplot.suptitle(
        "Silhouette analysis for DBSCAN clustering on trade data",
        fontsize=14, fontweight='bold')
    matplotlib.pyplot.plot(numpy.arange(x_min, x_max, 10), silhouette_avgs,
                           marker='o', color=
                           palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors[
                               1], markerfacecolor=
                           palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors[
                               2])
    matplotlib.pyplot.savefig('res/q3dbscan.png')
