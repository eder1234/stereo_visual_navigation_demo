def dijkstra(graph, start, end):
    """ 
    Dijkstra's algorithm implementation.
    """
    # The set of nodes already evaluated
    visited = set()
    # For each node, the cost of getting from the start node to that node.
    distances = [float('inf') for _ in range(len(graph))]
    distances[start] = 0
    # For each node, which node it can most efficiently be reached from.
    predecessors = [-1 for _ in range(len(graph))]
    
    while len(visited) < len(graph):
        # Select the node with the smallest distance value, from the set of nodes not yet visited
        current = min((value, idx) for idx, value in enumerate(distances) if idx not in visited)[1]
        
        # Found the destination
        if current == end:
            path = []
            while current != -1:
                path.append(current)
                current = predecessors[current]
            return path[::-1]
        
        visited.add(current)
        for neighbor, weight in enumerate(graph[current]):
            if weight == 0 or neighbor in visited:
                continue
            alt = distances[current] + weight
            if alt < distances[neighbor]:
                distances[neighbor] = alt
                predecessors[neighbor] = current

    return None  # Path not found

# Find paths using Dijkstra's algorithm on connectivity_graph
path_chess1_chess3_dijkstra = dijkstra(connectivity_graph, start_index_chess1_chess3, end_index_chess1_chess3)
path_artroom1_chess1_dijkstra = dijkstra(connectivity_graph, start_index_artroom1_chess1, end_index_artroom1_chess1)

path_chess1_chess3_dijkstra, path_artroom1_chess1_dijkstra
