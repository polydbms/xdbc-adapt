from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from experiments.model_optimizer.Configs import *


def process_data(data, training_data_per_env=-1, cluster_labes_avg=False, n_clusters=0):
    """
    This function precesses data to be used in the trasnfe rlearning algorithms.
    1. It groups the data for each combination of 'server_cpu', 'client_cpu' and 'network' into dataframes.
    2. For each of these dataframes it takes the first 'training_data_per_env' rows.
    3. It creates clusters of the individual dataframes based on their similarity.
    4. It groups the dataframes by their cluster, and creates a cluster_key containing all groups in that cluster.

    Parameters:
        data (dataframe): The Dataframe containing all datat.
        training_data_per_env (int): The amount of data to use per environment, -1 if all data should be used.

    Returns:
        Dict{cluster_key: dataframe}: Dictionary contaiing all clusterkeys and their corresponding dataframes.
    """

    # 1. Group the data
    grouped_data = defaultdict(list)
    df = data[(data['server_cpu'] > 0) & (data['client_cpu'] > 0) & (data['network'] > 0) & (data['run'] > 0)]
    for combination, group in df.groupby(['server_cpu', 'client_cpu', 'network']):
        if training_data_per_env != -1:
            # 2. take first n entries
            #if len(group) < training_data_per_env:
                #print(f"Length of data is less than specified data per environment ({len(group)} vs {training_data_per_env})")
            group = group.head(training_data_per_env)
        grouped_data[combination].append(group)
    grouped_data = {key: pd.concat(frames, ignore_index=True) for key, frames in grouped_data.items()}


    # 3. Create cluster
    clusters = _cluster_groups(grouped_data, column='uncompressed_throughput', n_clusters=n_clusters)

    grouped_df = pd.concat(grouped_data.values(), ignore_index=True)

    # 4. Group dataframes by their cluster
    group_to_rows = {}
    for group_key, group_df in grouped_data.items():
        group_to_rows[group_key] = grouped_df[
            (grouped_df['server_cpu'] == group_key[0]) &
            (grouped_df['client_cpu'] == group_key[1]) &
            (grouped_df['network'] == group_key[2])
            ]

    cluster_dataframes = {}
    for cluster_index, cluster in enumerate(clusters):
        # Create clusterkey containing all groups in that cluster
        cluster_key = "cluster-" + "-".join(f"{group_key[0]}_{group_key[1]}_{group_key[2]}" for group_key in cluster)

        averaged_environments = tuple(sum(elements) / len(cluster) for elements in zip(*cluster))
        #print(f"Cluster containing {cluster}")
        #print(f"Averaged to : {averaged_environments}")

        if cluster_labes_avg:
            cluster_key = f"cluster-avg_S{averaged_environments[0]}_C{averaged_environments[1]}_N{averaged_environments[2]}"

            #env_only = cluster_key.replace("cluster-avg_","")
            #env_dict = unparse_environment_float(env_only)

        # Combine all dataframes of that cluster
        cluster_rows = pd.concat([group_to_rows[group_key] for group_key in cluster], ignore_index=True)
        cluster_dataframes[cluster_key] = cluster_rows

    return cluster_dataframes


def _cluster_groups(grouped_data, column, n_clusters):
    """
    This function generates cluster of the grouped data.

    1. It calculates a similarity matrix based on a similarity function.
    2. It tries to find the optimal number of clusters using the silhouette score.
    3. Finaly it clusters the data into the specified number of clusters

    Parameters:
        grouped_data (dict): the grouped data to be clusters.
        column (str): the column name by which the similarity should be calculated.

    Returns:
        list: A list of all clusters with their cluster members.
    """
    group_keys = list(grouped_data.keys())
    group_frames = list(grouped_data.values())

    n = len(group_keys)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = _calculate_similarity(group_frames[i], group_frames[j], column)
            similarity_matrix[i, j] = distance
            similarity_matrix[j, i] = distance

    #optimal_clusters = _find_optimal_clusters_with_silhouette(similarity_matrix, max_clusters=len(group_keys) - 1)

    # -1 = no clusters
    # 0 = optimal number *2
    # else : custom number

    #if n_clusters == -1:
    #    actual_clusters = len(group_keys)  # min(optimal_clusters*2, len(group_keys) - 1) # todo optimal number of clusters seems too low ?
    #elif n_clusters == 0:
    #    actual_clusters = min((optimal_clusters), len(group_keys) - 1) #todo !!!! maybe limit how many different environments can be in the same cluster ?
    #else:
    #    actual_clusters = n_clusters

    #actual_clusters = int(len(group_keys))

    min_val = np.min(similarity_matrix)
    max_val = np.max(similarity_matrix)
    scaled_matrix = (similarity_matrix - min_val) / (max_val - min_val)

    distance_threshold = 0.15# orig = 0.15


    clustering = AgglomerativeClustering(n_clusters=None,
                                         affinity='precomputed',
                                         linkage='average',
                                         distance_threshold=distance_threshold)
    labels = clustering.fit_predict(scaled_matrix)

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(group_keys[i])

    return list(clusters.values())


def _calculate_similarity(group1, group2, column):
    """
    This function calculated the similarity between two dataframes based on the
    similarity of the distribution of values in the specified column.

    Parameters:
        group1 (dataframe): Data 1 to compare.
        group2 (dataframe): Data 2 to compare.
        column (str): Column on which to calculate the similarity on.

    Returns:
        float: The similarity of the two dataframe-columns.
    """
    values1 = group1[column].values
    values2 = group2[column].values

    #return wasserstein_distance(values1, values2)


    server_group_1 = group1['server_cpu'].unique()[0]
    client_group_1 = group1['client_cpu'].unique()[0]
    network_group_1 = transform_network(group1['network'].unique()[0])

    values_group_1 = np.array([server_group_1, client_group_1, network_group_1])

    server_group_2 = group2['server_cpu'].unique()[0]
    client_group_2 = group2['client_cpu'].unique()[0]
    network_group_2 = transform_network(group2['network'].unique()[0])

    values_group_2 = np.array([server_group_2, client_group_2, network_group_2])

    distances = np.linalg.norm(values_group_1 - values_group_2)

    return distances

def transform_network(x):
    return np.round(30 / (1 + 31 * np.exp(-0.02 * x)), 2)

def _find_optimal_clusters_with_silhouette(similarity_matrix, max_clusters=10):
    """
    This functino tries to find the optimal number of clusters for a similarity matrix.
    For that it creates a clustering for each possible number of clusters, calculates
    the silhouette-score for that clustering, and finaly return the number of clusters with
    the highest silhouette-score.

    Parameters:
        similarity_matrix (dataframe): Similarity matrix on which the clustering should be based.
        max_clusters (int): The maximum number of clusters to create.

    Returns:
        int: The number of optimal clusters.
    """
    silhouette_scores = []
    max_clusters = int(max_clusters)

    for n_clusters in range(2, max_clusters + 1):
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        labels = clustering.fit_predict(similarity_matrix)
        # Calculate silhouette score
        silhouette_avg = silhouette_score(similarity_matrix, labels, metric='precomputed')
        silhouette_scores.append(silhouette_avg)

    # Take clustering with best score
    optimal_clusters = np.argmax(silhouette_scores) + 2  # Add 2 because range starts at 2
    return optimal_clusters


def _group_data_by_environment(file_list):
    """
    Groups data by unique combinations of 'server_cpu', 'client_cpu', and 'network'.

    Parameters:
        file_list (list): List of file paths to load.

    Returns:
        dict: Grouped data with keys as combinations and values as concatenated DataFrames.
    """
    grouped_data = defaultdict(list)

    for file_path in file_list:
        df = pd.read_csv(file_path[0])
        df = df[(df['server_cpu'] > 0) & (df['client_cpu'] > 0) & (df['network'] > 0)]

        for combination, group in df.groupby(['server_cpu', 'client_cpu', 'network']):
            grouped_data[combination].append(group)

    return {key: pd.concat(frames, ignore_index=True) for key, frames in grouped_data.items()}