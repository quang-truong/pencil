import random
import networkx as nx
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque


def create_networkx_graph(graph_data):
    """Convert graph data to NetworkX graph"""
    G = nx.Graph()

    # Add edges (NetworkX will handle undirected edges automatically)
    for edge in graph_data["edges"]:
        src, target = edge
        G.add_edge(src, target)

    return G


def generate_random_graph(
    num_targets: int = 5, path_restricted: bool = True, return_neg_targets: bool = False
) -> Dict | None:
    """
    Generate a random undirected graph with the following properties:
    - Number of nodes in range [15, 30]
    - Extract edges (both directions for undirected graph)
    - Randomly pick root and target nodes
    - Create neighbor_k dictionary with k-hop neighbors

    Args:
        path_restricted: Whether to restrict neighbors to nodes on paths between root and target (default: True)
        return_neg_targets: Whether to return a dictionary of negative target nodes organized by distance from root (default: False)
    Returns:
        Dictionary containing:
        - edges: List of directed edges (both directions for undirected)
        - root: Randomly picked root node
        - targets: Randomly picked target nodes (2 <= distance <= 4 from root)
        - neighbor_k_dict: Dictionary of dictionaries with k-hop neighbors
        - num_nodes: Number of nodes in the graph
        - num_edges: Number of edges in the graph
        - neg_targets_dict: Dictionary where keys are distances from root and values are lists of nodes at
        those distances (if return_neg_targets=True)
    """
    # Generate random number of nodes between 15 and 30
    num_nodes = random.randint(15, 30)

    # Create a random undirected graph
    # Using Erdos-Renyi model with probability p to ensure connectivity
    p = 0.125  # Probability of edge creation
    G = nx.erdos_renyi_graph(num_nodes, p)

    # Ensure the graph is connected
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(num_nodes, p)

    # Convert to list of edges (both directions for undirected graph)
    edges = []
    for edge in G.edges():
        src, target = edge
        # enforce the edge to be in the form of [min(src, target), max(src, target)]
        edges.append([min(src, target), max(src, target)])

    # Randomly pick root node
    root = random.randint(0, num_nodes - 1)

    # Calculate shortest paths from root to all nodes
    shortest_paths = nx.single_source_shortest_path_length(G, root)

    # Find nodes with distance >= 2 from root
    valid_targets = [
        node
        for node, distance in shortest_paths.items()
        if distance >= 2 and distance <= 4 and node != root
    ]

    # If no valid targets, pick any node that's not the root
    if not valid_targets:
        valid_targets = [node for node in range(num_nodes) if node != root]

    # Randomly pick target nodes
    if len(valid_targets) < num_targets:
        # If not enough valid targets, use all available targets
        targets = valid_targets
    else:
        targets = list(random.sample(valid_targets, num_targets))

    # Create neighbor_k dictionary
    neighbor_k_dict = {}
    for target in targets:
        neighbor_k_dict[target] = create_neighbor_k_dict(
            G, root, target, path_restricted=path_restricted
        )

    if return_neg_targets:
        # Create a dictionary where each key is a distance from root
        # Negative targets of level 1 node would be nodes at distance 2 from root
        # Negative targets of level 2 node would be nodes at distance 3 from root
        neg_target_candidates: Dict[int, List[int]] = {i: [] for i in range(0, 5)}

        for node in range(num_nodes):
            try:
                distance = nx.shortest_path_length(G, root, node)
                if 4 >= distance - 1 >= 0:
                    neg_target_candidates[distance - 1].append(node)
            except nx.NetworkXNoPath:
                return None

        # If no negative unreachable targets (distance >= 5), return None
        if not neg_target_candidates[4]:
            return None

        return {
            "edges": edges,
            "root": root,
            "targets": targets,
            "neighbor_k_dict": neighbor_k_dict,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "neg_targets_dict": neg_target_candidates,
        }
    return {
        "edges": edges,
        "root": root,
        "targets": targets,
        "neighbor_k_dict": neighbor_k_dict,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "neg_targets_dict": None,
    }


def get_nodes_at_exact_distance(
    edge_list: List[Tuple[int, int]],
    root: int,
    distance: int,
    is_undirected: bool = True,
) -> List[int]:
    """
    Get nodes at exactly k-hop distance from root.

    Args:
        edge_list: List of edges
        root: Root node
        distance: Exact distance from root

    Returns:
        List of nodes at exactly the specified distance
    """
    if is_undirected:
        G = nx.Graph()
        G.add_edges_from(edge_list)
    else:
        G = nx.DiGraph()
        G.add_edges_from(edge_list)

    visited = {root}
    current_level = {root}

    # Process each level up to target distance
    for current_dist in range(distance):
        next_level = set()

        for node in current_level:
            for nbr in G.neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    next_level.add(nbr)

        if not next_level:
            return []  # No nodes at this distance

        current_level = next_level

    return list(current_level)


def create_neighbor_k_dict(
    G: nx.Graph, root: int, target: int, path_restricted: bool = True
) -> Dict[str, List[int]]:
    """
    Create neighbor_k dictionary with k-hop neighbors.

    Args:
        G: NetworkX graph
        root: Root node
        target: Target node
        path_restricted: Whether to restrict neighbors to nodes on paths between root and target (default: True)

    Returns:
        Dictionary with k-hop neighbors
    """
    # 1. Compute the true distance
    target_distance = nx.shortest_path_length(G, root, target)

    # 2. If path_restricted, find nodes on simple paths between root and target
    nodes_on_paths = None
    if path_restricted:
        try:
            shortest_path = nx.all_shortest_paths(G, root, target)
            nodes_on_paths = set()
            for path in shortest_path:
                nodes_on_paths.update(path)
        except nx.NetworkXNoPath:
            # If no path exists, return only root
            return {"0": [root]}

    # 3. Create neighbor_k dictionary
    neighbor_k: Dict[str, List[int]] = {"0": [root]}
    visited = {root}
    queue = deque([(root, 0)])

    # BFS to find k-hop neighbors
    while queue:
        node, dist = queue.popleft()

        # Don't go deeper once you're at the target layer
        if dist >= target_distance:
            continue

        for nbr in G.neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                nd = dist + 1

                # Only enqueue if within target_distance
                if nd <= target_distance:
                    # If path_restricted, also check if node is on paths
                    if not path_restricted or nbr in nodes_on_paths:
                        queue.append((nbr, nd))
                        neighbor_k.setdefault(str(nd), []).append(nbr)

    # 4. Sanity check
    max_k = max(map(int, neighbor_k.keys())) if neighbor_k else 0
    assert (
        max_k == target_distance
    ), f"largest layer = {max_k}, but expected {target_distance}"

    return neighbor_k


def generate_multiple_graphs(
    num_graphs: int = 10, path_restricted: bool = True, return_neg_targets: bool = True
) -> List[Dict]:
    """
    Generate multiple random graphs.

    Args:
        num_graphs: Number of graphs to generate

    Returns:
        List of graph dictionaries
    """
    graphs = []
    cnt = 0
    while cnt < num_graphs:
        graph_data = generate_random_graph(
            path_restricted=path_restricted, return_neg_targets=return_neg_targets
        )
        if graph_data is not None:
            graphs.append(graph_data)
            cnt += 1

    return graphs


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Generate a single graph
    graph: Dict | None = generate_random_graph()
    if graph is None:
        raise ValueError("Graph is None")
    else:
        print("Generated Graph:")
        print(f"Number of edges: {len(graph['edges'])}")
        print(f"Root: {graph['root']}")
        print(f"Target: {graph['target']}")
        print(f"Neighbor_k keys: {list(graph['neighbor_k'].keys())}")
        print(f"Neighbor_k: {graph['neighbor_k']}")

    # Generate multiple graphs
    print("\n" + "=" * 50)
    print("Generating multiple graphs...")
    multiple_graphs = generate_multiple_graphs(3)
    for i, graph in enumerate(multiple_graphs):
        print(f"\nGraph {i+1}:")
        print(f"  Root: {graph['root']}, Target: {graph['target']}")
        print(f"  Number of edges: {len(graph['edges'])}")
        print(f"  Neighbor_k levels: {list(graph['neighbor_k'].keys())}")
