import numpy as np

# Load the provided numpy array
matches_matrix = np.load('/mnt/data/matches_matrix.npy')

# Modify the matrix based on the conditions
connectivity_graph = np.where(matches_matrix > 100, 1, 0)

# Set the main diagonal to 1
np.fill_diagonal(connectivity_graph, 1)

# Node names
node_names = ["artroom1", "artroom2", "bandsaw1", "bandsaw2", "chess1", "chess2", "chess3", "podium1"]
