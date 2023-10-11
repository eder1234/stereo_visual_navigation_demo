from heapq import heappush, heappop

def heuristic(node1, node2, graph):
    """ 
    Heuristic function for A*.
    Returns the edge weight if it exists, otherwise a constant value (largest edge weight in the graph).
    """
    if graph[node1][node2] != 0:
        return graph[node1][node2]
    return np.max(graph)

def a_star_search(graph, start, end):
    """ 
    A* algorithm implementation.
    """
    # The set of nodes already evaluated
    closed_set = set()
    # The set of currently discovered nodes that are not evaluated yet.
    open_set = {start}
    # For each node, which node it can most efficiently be reached from.
    came_from = {}
    # For each node, the cost of getting from the start node to that node.
    g_score = {node: float('inf') for node in range(len(graph))}
    g_score[start] = 0
    # For each node, the total cost of getting from the start node to the goal
    # by passing by that node.
    f_score = {node: float('inf') for node in range(len(graph))}
    f_score[start] = heuristic(start, end, graph)
    
    # Priority queue for nodes to explore
    open_heap = [(f_score[start], start)]
    
    while open_set:
        # get the node in open_set having the lowest f_score[] value
        current = heappop(open_heap)[1]
        if current == end:
            # Path has been found
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        open_set.remove(current)
        closed_set.add(current)
        
        for neighbor in range(len(graph)):
            if graph[current][neighbor] == 0 or neighbor in closed_set:
                continue
            # d(current,neighbor) is the weight of the edge from current to neighbor
            tentative_g_score = g_score[current] + graph[current][neighbor]
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score[neighbor]:
                continue  # This is not a better path
            
            # This path is the best until now. Record it!
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end, graph)
            heappush(open_heap, (f_score[neighbor], neighbor))
    
    return None  # Path not found

# Node indices
start_index_chess1_chess3 = node_names.index("chess1")
end_index_chess1_chess3 = node_names.index("chess3")

start_index_artroom1_chess1 = node_names.index("artroom1")
end_index_artroom1_chess1 = node_names.index("chess1")

# Find paths using A* algorithm
path_chess1_chess3 = a_star_search(matches_matrix, start_index_chess1_chess3, end_index_chess1_chess3)
path_artroom1_chess1 = a_star_search(matches_matrix, start_index_artroom1_chess1, end_index_artroom1_chess1)

path_chess1_chess3, path_artroom1_chess1
