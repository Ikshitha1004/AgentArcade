# AgentArcade: AI Search & Optimization Algorithms on FrozenLake, TSP, and Chess

This repository implements classical search and optimization algorithms applied to three environments:

1. **FrozenLake** using:
   - Branch and Bound (BnB)
   - Iterative Deepening A* (IDA*)
   
2. **Travelling Salesman Problem (TSP)** using:
   - Simulated Annealing (SA)
   - Hill Climbing (HC)

3. **Chess** using:
   - K-ply Minimax
   - Alpha-Beta Pruning

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

---

## 2. Travelling Salesman Problem (TSP)

### Problem Description

The goal of the Travelling Salesman Problem (TSP) is to find the shortest possible route that visits each city exactly once and returns to the starting city.

- **Dataset Used**: `ch130.tsp` from [MicheleCattaneo/ant_colony_opt_TSP](https://github.com/MicheleCattaneo/ant_colony_opt_TSP)
- **Number of Cities**: 130
- **Distance Metric**: Euclidean distance between 2D city coordinates
- **Input Format**: Standard `.tsp` file (parsed into coordinates and a distance matrix)
- **Objective function**:

  f(tour) =<sub>  i=1</sub> Σ<sup> n</sup> distance(city<sub>i</sub>, city<sub>i+1</sub>) + distance(city<sub>n</sub>, city<sub>1</sub>)

---

### Implemented Algorithms

#### Simulated Annealing (SA)

- Probabilistic search that accepts worse solutions early on to escape local minima
- Temperature gradually decreases to reduce randomness over time

#### Hill Climbing (HC)

- Greedy local search using 2-opt swaps
- Continues improving until no better neighbor is found

---

## 3. Chess

### Environment Description

**Chess** is a classic two-player strategy game where one player controls the white pieces and the other controls the black pieces. The game is played on an 8x8 board with various pieces (king, queen, rook, bishop, knight, pawn) with specific movement rules.

- **Gym Environment**: `gym-chess`
- **Actions**: Moves based on chess rules
- **Goal**: Checkmate the opponent’s king or force a stalemate

### Implemented Algorithms

#### K-ply Minimax

- A depth-limited search algorithm that evaluates the game tree to a specified depth (k-ply).
- The agent selects the best possible move based on the minimax strategy, optimizing for the maximum score while minimizing the opponent’s chances.

#### Alpha-Beta Pruning

- A search algorithm that improves minimax by pruning branches that cannot influence the final decision.
- It enhances efficiency by ignoring branches that do not need to be explored due to the existence of better moves elsewhere.

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

#### FrozenLake Search Agents

Navigate to the `search_agents` folder and run:
```bash
cd search_agents
python main.py --agent bnb      # Branch and Bound
python main.py --agent ida     # IDA*
```

#### TSP Optimization Algorithms

Navigate to the `soln_agents` folder:
```bash
cd soln_agents
python main.py --agent sa      # Simulated Annealing
python main.py --agent hc      # Hill Climbing
```

#### Chess Game-Playing Agents

Navigate to the `game_playing_agents` folder and run:
```bash
cd game_playing_agents
python main.py --agent minimax     # K-ply Minimax
python main.py --agent alpha-beta  # Alpha-Beta Pruning
```

---

## PPT:

You can view or download the PowerPoint presentation from the link below:

[Presentation](https://github.com/Ikshitha1004/AgentArcade/blob/main/AI_ASSIGNMENT-2.pptx)

