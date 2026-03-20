import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn import cluster
from pyclustering.cluster.kmedoids import kmedoids

def evaluate_kmeans_elbow(data, k_range=range(2, 20), random_state=10):
    """
    Compute metrics for KMeans clustering evaluation
    """
    DATA = data.values

    KM = [KMeans(n_clusters=k, random_state=random_state).fit(DATA) for k in k_range]

    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(DATA, cent, 'euclidean') for cent in centroids]

    dist = [np.min(D, axis=1) for D in D_k]

    avgWithinSS = [np.sum(d) / DATA.shape[0] for d in dist]

    wcss = [np.sum(d**2) for d in dist]

    tss = np.sum((DATA - DATA.mean(0))**2)

    bss = [tss - w for w in wcss]

    return {
        "k_values": list(k_range),
        "avgWithinSS": avgWithinSS,
        "bss_ratio": [b / tss * 100 for b in bss]
    }

def evaluate_kmeans_silhouette(data, k_range=range(2, 20), random_state=10):
    """
    Compute silhouette scores for different values of K
    """
    DATA = data.values

    scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(DATA)

        score = silhouette_score(DATA, labels, metric='euclidean')
        scores.append(score)

    return {
        "k_values": list(k_range),
        "silhouette_scores": scores
    }

def run_clustering_algorithms(data, k_clusters=5):
    """
    Run multiple clustering algorithms on the dataset.

    Parameters:
    ----------
    data : array-like
        Dataset to cluster.
    k_clusters : int
        Number of clusters for KMeans and Hierarchical clustering.

    Returns:
    -------
    dict
        Dictionary with algorithm names as keys and cluster labels as values.
    """

    results = {}

    algorithms = {
        'kmeans': cluster.KMeans(n_clusters=k_clusters, random_state=10),
        'hierarchical': AgglomerativeClustering(n_clusters=k_clusters, linkage='ward'),
        'dbscan': DBSCAN(eps=0.6)
    }

    for name, model in algorithms.items():
        model.fit(data)
        results[name] = model.labels_

    return results

def compute_silhouette_scores(data, clustering_results):
    """
    Compute silhouette scores for multiple clustering results.

    Parameters:
    ----------
    data : array-like
    clustering_results : dict
        Output from clustering (algorithm_name -> labels)

    Returns:
    -------
    pd.DataFrame
    """

    metricas = []
    algorithms = []

    for name, labels in clustering_results.items():
        score = silhouette_score(data, labels, metric='euclidean')
        metricas.append(score)
        algorithms.append(name)

    resultados = pd.DataFrame({
        'algorithm': algorithms,
        'silhouette_score': metricas
    })

    return resultados

def run_kmedoids(data, initial_medoids, random_state=16):
    """
    Run K-Medoids clustering and return labels.

    Parameters:
    ----------
    data : array-like
    initial_medoids : list
    random_state : int

    Returns:
    -------
    list
        Cluster labels ordered according to input data
    """

    kmedoids_instance = kmedoids(data, initial_medoids)
    kmedoids_instance.process()

    clusters = kmedoids_instance.get_clusters()

    label_list = []

    for i, cluster in enumerate(clusters):
        df = pd.DataFrame(cluster, columns=['Index'])
        df['Cluster'] = i
        label_list.append(df)

    label_df = pd.concat(label_list, ignore_index=True)
    label_df = label_df.sort_values(by='Index')

    return label_df['Cluster'].values

def run_dbscan(data, eps=0.6):
    """
    Run DBSCAN clustering.

    Parameters:
    ----------
    data : array-like
    eps : float

    Returns:
    -------
    array
        Cluster labels
    """

    model = DBSCAN(eps=eps)
    model.fit(data)

    return model.labels_