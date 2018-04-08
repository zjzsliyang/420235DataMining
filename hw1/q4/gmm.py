import logging
import matplotlib.pyplot
from sklearn.cluster import DBSCAN, KMeans
import palettable.colorbrewer.qualitative
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# @profile
def gmm(df, eps, random_vip, knns):
    silhouette_avgs = []
    ks = []
    hits = []
    gmm_labels = []
    X = StandardScaler().fit_transform(df.T)
    for k in range(2, 12):
        clusterer = GaussianMixture(n_components=k, covariance_type='tied',
                                    max_iter=20, random_state=0).fit(X)
        cluster_labels = clusterer.predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_avgs.append(silhouette_avg)
        logging.info(
            "For n_clusters = %s ,the average silhouette_score in GMM is : %s." % (
            k, silhouette_avg))
        ks.append(k)
        gmm_labels.append(cluster_labels)

        hit = 0
        no = cluster_labels[df.columns.get_loc(random_vip)]
        for neighbor in knns:
            if cluster_labels[df.columns.get_loc(neighbor)] == no:
                hit += 1
            else:
                logging.debug(
                    "GMM: vipno: {} is not in the same cluster.".format(
                        neighbor))
        logging.info(
            "For k = {} in kNN, there has {} in the same cluster in GMM.".format(
                len(knns), hit))
        hits.append(hit)

    # plot_kmeans_clusterno(11, silhouette_avgs)

    # Compare with Kmeans
    kmeans_labels = KMeans(n_clusters=2, random_state=10).fit_predict(X)
    gmm_label = gmm_labels[ks.index(2)]
    hit = 0
    for index, kmeans_label in enumerate(kmeans_labels):
        if kmeans_label == gmm_label[index]:
            hit += 1
    logging.info(
        "The accuracy of KMeans is {}".format(hit / len(kmeans_labels)))

    # Compare with DBSCAN
    dbscan_labels = DBSCAN(eps, min_samples=10).fit_predict(X)
    gmm_label = gmm_labels[ks.index(len(set(dbscan_labels)))]
    dbscan_labels[dbscan_labels == -1] = 1
    hit = 0
    for index, dbscan_label in enumerate(dbscan_labels):
        if dbscan_label == gmm_label[index]:
            hit += 1
    logging.info(
        "The accuracy of DBSCAN is {}".format(hit / len(dbscan_labels)))


def plot_kmeans_clusterno(k, silhouette_avgs):
    matplotlib.pyplot.figure(figsize=(10, 6))
    matplotlib.pyplot.xlim([2, k + 1])
    matplotlib.pyplot.ylim([-0.3, 0.2])
    matplotlib.pyplot.xlabel(
        "The number of clusters as well as the number of centroids")
    matplotlib.pyplot.ylabel("The silhouette coefficient values")
    matplotlib.pyplot.title(
        "Silhouette analysis for GMM clustering on trade data",
        fontsize=14, fontweight='bold', y=1.08)
    matplotlib.pyplot.plot(range(2, k + 2), silhouette_avgs, marker='o', color=
    palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors[1], markerfacecolor=
                           palettable.colorbrewer.qualitative.Pastel1_4.mpl_colors[
                               2])
    matplotlib.pyplot.savefig('res/q4gmm.png')
