# Topological map

This folder contains the files related with the topological map (visual memory).

## Overview
Point features are extracted and matched (SuperGlue) between all the pairs of two images in a folder (2021 mobile stereo dataset).

### Search path algorithms

Weighted graph (`matches_matrix`) used by A*.

|              | artroom1 | artroom2 | bandsaw1 | bandsaw2 | chess1 | chess2 | chess3 | podium1 |
|--------------|----------|----------|----------|----------|--------|--------|--------|---------|
| **artroom1** | 0        | 264      | 0        | 0        | 1      | 5      | 4      | 0       |
| **artroom2** | 264      | 0        | 0        | 0        | 0      | 6      | 0      | 0       |
| **bandsaw1** | 0        | 0        | 0        | 86       | 0      | 0      | 0      | 0       |
| **bandsaw2** | 0        | 0        | 86       | 0        | 0      | 2      | 0      | 0       |
| **chess1**   | 1        | 0        | 0        | 0        | 0      | 193    | 165    | 0       |
| **chess2**   | 5        | 6        | 0        | 2        | 193    | 0      | 206    | 0       |
| **chess3**   | 4        | 0        | 0        | 0        | 165    | 206    | 0      | 0       |
| **podium1**  | 0        | 0        | 0        | 0        | 0      | 0      | 0      | 0       |

Shortest paths:
* From "chess1" to "chess3": chess1 → artroom1 → chess3
* From "artroom1" to "chess1": artroom1 → chess1

Connectivity graph used by Dijkstra's algorithm.

|              | artroom1 | artroom2 | bandsaw1 | bandsaw2 | chess1 | chess2 | chess3 | podium1 |
|--------------|----------|----------|----------|----------|--------|--------|--------|---------|
| **artroom1** | 1        | 1        | 0        | 0        | 0      | 0      | 0      | 0       |
| **artroom2** | 1        | 1        | 0        | 0        | 0      | 0      | 0      | 0       |
| **bandsaw1** | 0        | 0        | 1        | 0        | 0      | 0      | 0      | 0       |
| **bandsaw2** | 0        | 0        | 0        | 1        | 0      | 0      | 0      | 0       |
| **chess1**   | 0        | 0        | 0        | 0        | 1      | 1      | 1      | 0       |
| **chess2**   | 0        | 0        | 0        | 0        | 1      | 1      | 1      | 0       |
| **chess3**   | 0        | 0        | 0        | 0        | 1      | 1      | 1      | 0       |
| **podium1**  | 0        | 0        | 0        | 0        | 0      | 0      | 0      | 1       |

Shortest paths:    
* From "chess1" to "chess3": chess1 → chess3
* From "artroom1" to "chess1": No path found.
