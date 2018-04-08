from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def gmm(df, k, eps, random_vip, knns):
    # TODO: calculate the accuracy
    X = StandardScaler().fit_transform(df.T)
    # Compare with KMeans
    gmix_kmeans = GaussianMixture(n_components=k)
    gmix_kmeans.fit(X)
    print(gmix_kmeans.means_)

    # Compare with DBSCAN
    gmix_dbscan = GaussianMixture(n_components=eps)
    gmix_dbscan.fit(X)
    print(gmix_dbscan.means_)

    # Compare with kNN