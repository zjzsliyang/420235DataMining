import math
import numpy
import logging
import matplotlib.pyplot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import palettable.colorbrewer.qualitative
from sklearn.metrics import silhouette_score, silhouette_samples


# @profile
def kmeans(df, random_vip, knns):
    k = int(math.sqrt(df.shape[1] / 2))
    silhouette_avgs = []

    for n_clusters in range(2, k + 2):
        logging.debug("K-means: n_clusters = {}".format(n_clusters))
        clusterer = KMeans(n_clusters=n_clusters)
        X = PCA(n_components=2, whiten=True).fit_transform(df.T)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avgs.append(silhouette_avg)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score in K-means is :", silhouette_avg)

        if n_clusters >= k / 2:
            plot_silhouette(X, cluster_labels, n_clusters, clusterer)

        res = 0
        no = cluster_labels[df.columns.get_loc(random_vip)]
        for neighbor in knns:
            if cluster_labels[df.columns.get_loc(neighbor)] == no:
                res += 1
            else:
                logging.info(
                    "K-means: vipno: {} is not in the same cluster.".format(
                        neighbor))
        print(
            "For k = {} in kNN, there has {} in the same cluster in K-means.".format(
                len(knns), res))

    plot_kmeans_clusterno(k, silhouette_avgs)

    return silhouette_avgs.index(max(silhouette_avgs)) + 2


def plot_kmeans_clusterno(k, silhouette_avgs):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.xlim([2, k + 1])
    matplotlib.pyplot.ylim([-0.1, 1])
    matplotlib.pyplot.xlabel(
        "The number of clusters as well as the number of centroids")
    matplotlib.pyplot.ylabel("The silhouette coefficient values")
    matplotlib.pyplot.title(
        "Silhouette analysis for K-means clustering on trade data",
        fontsize=14, fontweight='bold', y=1.08)
    matplotlib.pyplot.plot(range(2, k + 2), silhouette_avgs, marker='o', color=
    palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors[1], markerfacecolor=
                           palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors[
                               2])
    matplotlib.pyplot.savefig('res/q2kmeans.png')


def plot_silhouette(X, cluster_labels, n_clusters, clusterer):
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    colors = palettable.colorbrewer.qualitative.Set3_12.mpl_colors
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = colors[i % 12]
        ax1.fill_betweenx(numpy.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    matplotlib.pyplot.suptitle(
        ("Silhouette analysis for KMeans clustering on sample data "
         "with n_clusters = %d" % n_clusters),
        fontsize=14, fontweight='bold')
    matplotlib.pyplot.savefig('res/q2silhouette%d.png' % n_clusters)
