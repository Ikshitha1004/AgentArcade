# AgentArcade: AI Search & Optimization Algorithms on FrozenLake and TSP

This repository implements four classical search and optimization algorithms applied to two environments:

1. **FrozenLake** using:
   - Branch and Bound (BnB)
   - Iterative Deepening A* (IDA*)
2. **Travelling Salesman Problem (TSP)** using:
   - Simulated Annealing (SA)
   - Hill Climbing (HC)

---

## 1. FrozenLake 

### Environment Description

**FrozenLake-v1** is a grid-based navigation environment where an agent must move from a starting point `S` to a goal `G`, avoiding dangerous holes `H` and walking safely on frozen tiles `F`.

Example 4x4 Map:
```
  S F F F
  H F H F
  F F F H
  F H F G
```

- **Gym Environment**: `FrozenLake-v1` from `gymnasium`
- **Mode**: `is_slippery=False` is used to make the environment deterministic
- **Actions**: Up, Down, Left, Right
- **Goal**: Safely reach the goal tile `G` from the start `S`

### Implemented Algorithms

#### Branch and Bound (BnB)

- Uses depth-first search with cost-based pruning
- Efficiently eliminates paths that exceed known optimal cost

#### Iterative Deepening A* (IDA*)

- Combines DFS with a cost threshold (`f = g + h`)
- Uses Manhattan Distance as a heuristic
- Optimal for deterministic environments


## 2. Travelling Salesman Problem (TSP)

### Problem Description

The goal of the Travelling Salesman Problem (TSP) is to find the shortest possible route that visits each city exactly once and returns to the starting city.

- **Dataset Used**: `ch130.tsp` from [MicheleCattaneo/ant_colony_opt_TSP](https://github.com/MicheleCattaneo/ant_colony_opt_TSP)
- **Number of Cities**: 130
- **Distance Metric**: Euclidean distance between 2D city coordinates
- **Input Format**: Standard `.tsp` file (parsed into coordinates and a distance matrix)

---

### Implemented Algorithms

#### Simulated Annealing (SA)

- Probabilistic search that accepts worse solutions early on to escape local minima
- Temperature gradually decreases to reduce randomness over time

#### Hill Climbing (HC)

- Greedy local search using 2-opt swaps
- Continues improving until no better neighbor is found

---


## Installation

Follow these steps to set up the AgentArcade environment:

### 1. Clone the Repository
```
git clone https://github.com/Ikshitha1004/AgentArcade.git
cd AgentArcade
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run Algorithms
Execute any algorithm script:

# Simulated Annealing (TSP Optimization)
```
cd soln_agents
python run_SA.py
```

# Branch and Bound

```
cd search_agents
python run_BnB.py
```


