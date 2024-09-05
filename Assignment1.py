import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import sys

# Increase the recursion limit to handle large datasets
sys.setrecursionlimit(10000)

# Global adjacency list for managing the graph structure
global_adjacency_list = defaultdict(set)

def renumber_nodes(edges, max_nodes=500):
    """
    Renumber nodes in the edges list to a consecutive range starting from 0.
    The number of nodes can be limited by `max_nodes`.
    """
    all_nodes = np.concatenate(edges)
    unique_nodes = np.unique(all_nodes)
    
    if max_nodes is None:
        max_nodes = len(unique_nodes)
    
    max_nodes = min(max_nodes, len(unique_nodes))
    unique_nodes = unique_nodes[:max_nodes]
    
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    renumbered_edges = [[node_mapping.get(src, -1), node_mapping.get(dst, -1)] for src, dst in edges]
    
    # Filter out edges that have any nodes not in the node_mapping
    renumbered_edges = [edge for edge in renumbered_edges if -1 not in edge]
   
    return np.array(renumbered_edges), len(unique_nodes)

def import_lastfm_asia_data(file_path):
    """
    Import LastFM Asia dataset, renumber nodes, and remove duplicate edges.
    """
    edges = []
    with open(file_path, 'r') as file:
        # Skip header line if present
        file.readline()  # Skip the header
        
        for line in file:
            src, dst = map(int, line.strip().split(','))
            edges.append([src, dst])
    
    edges_array = np.array(edges)
    unique_edges_array = np.unique(edges_array, axis=0)
    renumbered_edges, num_nodes = renumber_nodes(unique_edges_array)
    
    return renumbered_edges, num_nodes

def import_wiki_vote_data(file_path):
    """
    Import Wiki-Vote dataset, renumber nodes, and remove duplicate edges.
    """
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comments
            src, dst = map(int, line.strip().split())
            edges.append([src, dst])
    
    edges_array = np.array(edges)
    unique_edges_array = np.unique(edges_array, axis=0)
    renumbered_edges, num_nodes = renumber_nodes(unique_edges_array)
    
    return renumbered_edges, num_nodes

def prepare_adjacency_list(edges):
    """
    Prepare an adjacency list from the edge list.
    """
    global global_adjacency_list
    global_adjacency_list = defaultdict(set)
    
    for src, dst in edges:
        global_adjacency_list[src].add(dst)
        global_adjacency_list[dst].add(src)  # Ensure the graph is undirected
    
    return global_adjacency_list

def single_source_betweenness(adjacency_list, source, num_nodes):
    """
    Calculate betweenness centrality for a single source node.
    """
    betweenness = {}
    stack = []
    predecessors = {i: [] for i in range(num_nodes)}
    sigma = np.zeros(num_nodes)
    sigma[source] = 1
    distance = [-1] * num_nodes
    distance[source] = 0
    queue = [source]

    while queue:
        v = queue.pop(0)
        stack.append(v)
        for w in adjacency_list[v]:
            if distance[w] < 0:
                queue.append(w)
                distance[w] = distance[v] + 1
            if distance[w] == distance[v] + 1:
                sigma[w] += sigma[v]
                predecessors[w].append(v)

    delta = np.zeros(num_nodes)
    while stack:
        w = stack.pop()
        for v in predecessors[w]:
            c = (sigma[v] / sigma[w]) * (1 + delta[w])
            edge = tuple(sorted((v, w)))
            betweenness[edge] = betweenness.get(edge, 0) + c
            delta[v] += c

    return betweenness

def compute_edge_betweenness_centrality(adjacency_list, num_nodes):
    """
    Compute the edge betweenness centrality for all edges in the graph.
    """
    edge_betweenness = {}
    
    for node in range(num_nodes):
        betweenness = single_source_betweenness(adjacency_list, node, num_nodes)
        for edge, value in betweenness.items():
            edge_betweenness[edge] = edge_betweenness.get(edge, 0) + value

    print("Edge Betweenness Computation Complete")
    return edge_betweenness

def remove_edge_with_highest_betweenness(edge_betweenness):
    """
    Remove the edge with the highest betweenness centrality from the graph.
    """
    global global_adjacency_list
    sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_edges:
        edge_to_remove = sorted_edges[0][0]
        src, dst = edge_to_remove
        print(f"Removing edge: {src}-{dst}")
        global_adjacency_list[src].discard(dst)
        global_adjacency_list[dst].discard(src)
    
    return global_adjacency_list

def find_connected_components(num_nodes):
    """
    Find all connected components in the graph.
    """
    global global_adjacency_list
    visited = [False] * num_nodes
    components = []
    
    def dfs(node, component):
        stack = [node]
        while stack:
            curr = stack.pop()
            if not visited[curr]:
                visited[curr] = True
                component.append(curr)
                for neighbor in global_adjacency_list[curr]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
    
    for node in range(num_nodes):
        if not visited[node] and node in global_adjacency_list:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components

def assign_communities(components, num_nodes):
    """
    Assign community IDs to nodes based on their connected components.
    """
    community_ids = np.zeros(num_nodes, dtype=int)
    for component in components:
        min_node_id = min(component)
        for node in component:
            community_ids[node] = min_node_id
    
    return community_ids

def Girvan_Newman_one_level(num_nodes):
    """
    Perform one level of the Girvan-Newman algorithm to identify communities.
    """
    edge_betweenness = compute_edge_betweenness_centrality(global_adjacency_list, num_nodes)
    remove_edge_with_highest_betweenness(edge_betweenness)
    components = find_connected_components(num_nodes)
    print(f"Number of components: {len(components)}")
    community_ids = assign_communities(components, num_nodes)
    return community_ids, components

def calculate_modularity(components, adjacency_list, m):
    """
    Calculate the modularity of the current partitioning of the graph.
    """
    modularity = 0.0
    for component in components:
        for node1 in component:
            for node2 in component:
                A = 1 if node2 in adjacency_list[node1] else 0
                k1 = len(adjacency_list[node1])
                k2 = len(adjacency_list[node2])
                modularity += (A - (k1 * k2) / (2 * m))
    return modularity / (2 * m)

def Girvan_Newman(nodes_connectivity_list, num_nodes):
    """
    Perform Girvan-Newman algorithm to identify communities.
    """
    community_mat = []
    previous_modularity = None
    best_modularity = -float('inf')
    best_partition = None
    m = len(nodes_connectivity_list)
    increasing = None  # To track if modularity is increasing or decreasing

    while True:
        current_partition, components = Girvan_Newman_one_level(num_nodes)
        modularity = calculate_modularity(components, global_adjacency_list, m)
        print(f"Modularity: {modularity}")

        # If this is the first iteration, just update the previous_modularity and continue
        if previous_modularity is None:
            previous_modularity = modularity
            best_modularity = modularity
            best_partition = current_partition
            community_mat = current_partition
            continue

        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = current_partition
            community_mat = current_partition

            if increasing is False:  # If modularity was decreasing and now increases, stop at the valley
                print("Modularity increased after decreasing. Stopping at valley.")
                break
            increasing = True  # Modularity is increasing

        else:
            if increasing:  # If modularity was increasing and now decreases, stop at the peak
                print("Modularity decreased after increasing. Stopping at peak.")
                break
            increasing = False  # Modularity is decreasing

        # Check for modularity in the 0.3 to 0.7 range and stop if it's within this range
        if 0.3 <= modularity <= 0.7:
            print("Modularity is within the acceptable range (0.4 to 0.6).")
            break

        previous_modularity = modularity  # Update the previous modularity

    
    community_mat = np.array(best_partition).reshape(-1, 1)
    return community_mat


def create_distance_matrix(community_mat):
    """
    Create a distance matrix for dendrogram visualization based on community assignments.
    """
    num_nodes = len(community_mat)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    
    # Assign distances based on community assignments
    for i in range(num_nodes):
        for j in range(num_nodes):
            distance_matrix[i, j] = 0 if community_mat[i] == community_mat[j] else 1
    
    return distance_matrix

def calculate_linkage_matrix(distance_matrix):
    """
    Create a linkage matrix using the complete linkage method manually.
    """
    num_points = distance_matrix.shape[0]
    clusters = {i: [i] for i in range(num_points)}
    linkage_matrix = []

    current_cluster_id = num_points

    while len(clusters) > 1:
        min_dist = float('inf')
        pair_to_merge = None
        
        for i in clusters:
            for j in clusters:
                if i >= j:
                    continue
                distances = [distance_matrix[p1][p2] for p1 in clusters[i] for p2 in clusters[j]]
                max_dist = max(distances)
                
                if max_dist < min_dist:
                    min_dist = max_dist
                    pair_to_merge = (i, j)

        i, j = pair_to_merge
        new_cluster = clusters[i] + clusters[j]
        linkage_matrix.append([i, j, min_dist, len(new_cluster)])
        
        del clusters[i]
        del clusters[j]
        clusters[current_cluster_id] = new_cluster
        current_cluster_id += 1

    return np.array(linkage_matrix)

def visualize_dendrogram(distance_matrix, dataset):
    """
    Visualize and save a dendrogram based on the distance matrix.
    """
    # condensed_distance_matrix = squareform(distance_matrix, checks=False)
    Z = calculate_linkage_matrix(distance_matrix)

    plt.figure(figsize=(10, 7))
    plt.title(f"Dendrogram for {dataset} Dataset")
    dendrogram(Z)
    plt.xlabel('Nodes')
    plt.ylabel('Distance')
    # plt.show()
    plt.savefig(f"dendrogram_{dataset}.png")

def create_community_matrix(partition, num_nodes):
    community_mat = np.zeros((num_nodes, 1))
    for node, community in enumerate(partition):
        community_mat[node, 0] = community
    return community_mat

def louvain_one_iter(nodes_connectivity_list, num_nodes):
    """
    Perform one level of the Louvain algorithm to identify communities.
    """
    def modularity_gain(node, community, community_sum_degrees, node_degree, m):
        # Calculate the modularity gain when moving 'node' to 'community'
        sigma_tot = community_sum_degrees
        ki = node_degree
        ai = ki / (2 * m)
        bi = sigma_tot / (2 * m)
        return ai - bi * bi

    # Create the adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in nodes_connectivity_list:
        adjacency_matrix[edge[0]][edge[1]] = 1
        adjacency_matrix[edge[1]][edge[0]] = 1

    degree = np.sum(adjacency_matrix, axis=1)
    m = np.sum(degree) / 2

    # Initialize communities
    communities = list(range(num_nodes))
    best_partition = communities.copy()
    modularity = -1
    improvement = True
    
    while improvement:
        improvement = False
        for node in range(num_nodes):
            current_community = communities[node]
            best_community = current_community
            max_gain = 0

            community_sum_degrees = {community: np.sum(degree[communities == community]) for community in np.unique(communities)}

            for neighbor in range(num_nodes):
                if adjacency_matrix[node, neighbor] == 1:
                    neighbor_community = communities[neighbor]
                    if neighbor_community != current_community:
                        gain = modularity_gain(node, neighbor_community, community_sum_degrees[neighbor_community], degree[node], m)
                        if gain > max_gain:
                            max_gain = gain
                            best_community = neighbor_community

            if best_community != current_community:
                communities[node] = best_community
                improvement = True

        # Update modularity
        new_modularity = 0
        for community in np.unique(communities):
            nodes_in_community = np.where(communities == community)[0]
            for i in nodes_in_community:
                for j in nodes_in_community:
                    new_modularity += adjacency_matrix[i, j] - (degree[i] * degree[j]) / (2 * m)
        new_modularity /= (2 * m)

        if new_modularity > modularity:
            modularity = new_modularity
            best_partition = communities.copy()
        else:
            break
    # graph_partition = np.array([best_partition[node] for node in range(num_nodes)])
    
    return best_partition

def save_partition_to_file(graph_partition, file_path):
    """
    Save Louvain partition to file
    """
    np.savetxt(file_path, graph_partition, fmt='%d', delimiter='\n')
    print(f"Graph partition saved to {file_path}")

if __name__ == "__main__":

    ############ Answer qn 1-4 for wiki-vote data #################################################
    # Import wiki-vote.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.

    nodes_connectivity_list_wiki,num_nodes_wiki = import_wiki_vote_data('../data/wiki-Vote.txt')
    prepare_adjacency_list(nodes_connectivity_list_wiki)
    
    # This is for question no. 1
    # graph_partition: graph_partitition is a nx1 numpy array where the rows corresponds to nodes in the network (0 to n-1) and
    #                  the elements of the array are the community ids of the corressponding nodes.
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    graph_partition_wiki  = Girvan_Newman_one_level(num_nodes_wiki)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # community_mat is a n x m matrix, where m is the number of levels of Girvan-Newmann algorithm and n is the number of nodes in the network.
    # Columns of the matrix corresponds to the graph_partition which is a nx1 numpy array, as before, corresponding to each level of the algorithm. 
    wiki_community_mat = Girvan_Newman(nodes_connectivity_list_wiki,num_nodes_wiki)
    np.savetxt('community_mat_wiki_gm.csv', wiki_community_mat, delimiter='\n', fmt='%d')

    # This is for question no. 3
    # Visualise dendogram for the communities obtained in question no. 2.
    # Save the dendogram as a .png file in the current directory.
    wiki_distance_matrix = create_distance_matrix(wiki_community_mat)
    visualize_dendrogram(wiki_distance_matrix, dataset="Wiki-Vote_gm")

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before. Show the resulting communities after one iteration of the algorithm.
    graph_partition_louvain_wiki = louvain_one_iter(nodes_connectivity_list_wiki,num_nodes_wiki)
    save_partition_to_file(graph_partition_louvain_wiki, 'graph_partition_louvain_wiki.txt')
    community_mat_wiki = create_community_matrix(graph_partition_louvain_wiki, num_nodes_wiki)
    wiki_distance_matrix = create_distance_matrix(community_mat_wiki)

    visualize_dendrogram(wiki_distance_matrix, "Wiki-Vote_louvain")
    

    # ############ Answer qn 1-4 for bitcoin data #################################################
    # Import lastfm_asia_edges.csv
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.

    # global_adjacency_list.clear()
    # Question 1
    nodes_connectivity_list_lastfm, num_nodes_lastfm = import_lastfm_asia_data("../data/lastfm_asia_edges.csv")
    prepare_adjacency_list(nodes_connectivity_list_lastfm)
    

    # Question 2
    graph_partition_lastfm  = Girvan_Newman_one_level(num_nodes_lastfm)
    lastfm_community_mat = Girvan_Newman(nodes_connectivity_list_lastfm, num_nodes_lastfm)
    np.savetxt('community_mat_last_fm_gm.csv', lastfm_community_mat, delimiter='\n', fmt='%d')

    # Question 3
    lastfm_distance_matrix = create_distance_matrix(lastfm_community_mat)
    visualize_dendrogram(lastfm_distance_matrix, dataset="Last-FM_gm")

    # Question 4
    graph_partition_louvain_lastfm = louvain_one_iter(nodes_connectivity_list_lastfm,num_nodes_lastfm)
    save_partition_to_file(graph_partition_louvain_lastfm, 'graph_partition_louvain_lastfm.txt')
    community_mat_last_fm = create_community_matrix(graph_partition_louvain_lastfm, num_nodes_lastfm)
    lastfm_distance_matrix = create_distance_matrix(community_mat_last_fm)

    visualize_dendrogram(lastfm_distance_matrix, "LastFM_louvain")

