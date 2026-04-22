"""
================================================================================
Fair and Efficient Task Allocation and Path Planning for Multi-Agent Systems
in Disaster Response Using Linear Programming
================================================================================
Author: Trung Hieu Pham (Jack Pham)
Research: NSF REU Site: CI Research 4 Social Change, Award #2150390
Advisors: Dr. Adam Thorpe, Dr. Ufuk Topcu
Institution: Cypress College / University of Texas at Austin

This module implements:
  - Grid world environment (MDP formulation)
  - FE-MADDPG: Fairness-Enhanced Multi-Agent Deep Deterministic Policy Gradient
  - CBS: Conflict-Based Search for collision-free pathfinding
  - LP Integration: Linear Programming for task allocation optimization
  - Baseline greedy algorithm for comparison
  - Visualization and metrics collection

Usage:
  from disaster_response_sim import DisasterEnv, run_experiment
  results = run_experiment(algorithm='LP_CBS', seed=42)
================================================================================
"""

import numpy as np
import random
import heapq
import copy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.optimize import linprog
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

AGENT_COLORS = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'red', 'lime']
CELL_FREE     = 0
CELL_OBSTACLE = 1
CELL_FIRE     = 2
CELL_AGENT    = 3

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: int
    position: Tuple[int, int]
    completed: bool = False
    assigned_to: int = -1

@dataclass
class Agent:
    id: int
    position: Tuple[int, int]
    start: Tuple[int, int] = field(init=False)
    path: List[Tuple[int, int]] = field(default_factory=list)
    tasks_completed: int = 0
    total_distance: float = 0.0
    color: str = 'blue'

    def __post_init__(self):
        self.start = self.position

@dataclass
class CBSNode:
    cost: float
    constraints: List[Tuple]   # (agent_id, position, timestep)
    paths: Dict[int, List]
    conflicts: List[Tuple] = field(default_factory=list)

    def __lt__(self, other):
        return self.cost < other.cost

# ─────────────────────────────────────────────────────────────────────────────
# GRID WORLD ENVIRONMENT  (MDP Formulation)
# ─────────────────────────────────────────────────────────────────────────────

class DisasterEnv:
    """
    2D Grid World MDP for multi-agent disaster response.

    State  S : grid snapshot + agent positions + task statuses
    Action A : movement choices derived from LP / CBS / Greedy
    Reward R : fairness reward r_t^i = (ε + |e_t^i / ē_t - 1|) / ē_t
    Policy π : dictated by chosen algorithm
    γ        : discount factor for future rewards
    """

    def __init__(self,
                 grid_size: int = 10,
                 n_agents: int = 5,
                 n_tasks: int = 10,
                 n_obstacles: int = 10,
                 seed: int = 42,
                 epsilon: float = 1e-6,
                 gamma: float = 0.95):

        self.grid_size   = grid_size
        self.n_agents    = n_agents
        self.n_tasks     = n_tasks
        self.n_obstacles = n_obstacles
        self.seed        = seed
        self.epsilon     = epsilon   # ε to prevent division by zero
        self.gamma       = gamma     # discount factor

        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.grid:    np.ndarray = None
        self.agents:  List[Agent] = []
        self.tasks:   List[Task]  = []
        self.episode_rewards: Dict[int, List[float]] = defaultdict(list)
        self.metrics: Dict[str, Any] = {}

        self._build_world()

    # ── World construction ──────────────────────────────────────────────────

    def _build_world(self):
        """Deterministically build grid using seeded RNG."""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        occupied = set()

        # Place obstacles
        while len(occupied) < self.n_obstacles:
            r, c = self.rng.integers(0, self.grid_size, size=2)
            occupied.add((r, c))
            self.grid[r, c] = CELL_OBSTACLE

        # Place agents
        self.agents = []
        for i in range(self.n_agents):
            while True:
                r, c = self.rng.integers(0, self.grid_size, size=2)
                if (r, c) not in occupied:
                    occupied.add((r, c))
                    color = AGENT_COLORS[i % len(AGENT_COLORS)]
                    self.agents.append(Agent(id=i, position=(r, c), color=color))
                    break

        # Place fire tasks
        self.tasks = []
        for i in range(self.n_tasks):
            while True:
                r, c = self.rng.integers(0, self.grid_size, size=2)
                if (r, c) not in occupied:
                    occupied.add((r, c))
                    self.tasks.append(Task(id=i, position=(r, c)))
                    self.grid[r, c] = CELL_FIRE
                    break

    def reset(self):
        """Reset environment to initial state (keeps same seed-based layout)."""
        self._build_world()
        self.episode_rewards = defaultdict(list)
        return self._get_state()

    def _get_state(self) -> Dict:
        return {
            'grid': self.grid.copy(),
            'agent_positions': [a.position for a in self.agents],
            'task_positions':  [t.position for t in self.tasks],
            'task_completed':  [t.completed for t in self.tasks],
        }

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return (0 <= r < self.grid_size and
                0 <= c < self.grid_size and
                self.grid[r, c] != CELL_OBSTACLE)

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid neighboring cells (4-directional + wait)."""
        r, c = pos
        moves = [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r, c)]  # last = wait
        return [p for p in moves if self.is_valid(p)]

    # ── Fairness Reward  (Eq. from paper) ───────────────────────────────────

    def fairness_reward(self, agent_id: int, performances: List[float]) -> float:
        """
        r_t^i = (ε + |e_t^i / ē_t - 1|) / ē_t

        performances : list of e_t^i for all agents
                       (inverse distance to assigned task — lower dist = better perf)
        """
        e_i   = performances[agent_id]
        e_bar = np.mean(performances) if np.mean(performances) != 0 else self.epsilon
        reward = (self.epsilon + abs(e_i / e_bar - 1)) / e_bar
        return float(reward)

    def compute_performance(self, agent: Agent, task: Task) -> float:
        """e_t^i : performance as inverse Manhattan distance to task."""
        dist = abs(agent.position[0] - task.position[0]) + \
               abs(agent.position[1] - task.position[1])
        return 1.0 / (dist + self.epsilon)

    def step(self, agent_id: int, new_pos: Tuple[int, int],
             performances: List[float]) -> Tuple[Dict, float, bool]:
        """Move agent, collect reward, check task completion."""
        agent = self.agents[agent_id]
        if self.is_valid(new_pos):
            old_pos = agent.position
            agent.total_distance += abs(new_pos[0]-old_pos[0]) + abs(new_pos[1]-old_pos[1])
            agent.position = new_pos

        reward = self.fairness_reward(agent_id, performances)
        self.episode_rewards[agent_id].append(reward)

        # Check task completion
        done = False
        for task in self.tasks:
            if not task.completed and task.position == agent.position:
                task.completed = True
                agent.tasks_completed += 1

        done = all(t.completed for t in self.tasks)
        return self._get_state(), reward, done


# ─────────────────────────────────────────────────────────────────────────────
# A* SINGLE-AGENT PATHFINDING
# ─────────────────────────────────────────────────────────────────────────────

def manhattan(a: Tuple, b: Tuple) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(env: DisasterEnv, start: Tuple, goal: Tuple,
          constraints: set = None,
          max_time: int = 200) -> Optional[List[Tuple]]:
    """
    A* with optional space-time constraints for CBS.
    constraints: set of (position, timestep) pairs that are forbidden.
    """
    if constraints is None:
        constraints = set()

    open_list = []
    heapq.heappush(open_list, (manhattan(start, goal), 0, start, [start]))
    visited = {}

    while open_list:
        f, t, pos, path = heapq.heappop(open_list)

        if pos == goal:
            return path

        if t >= max_time:
            continue

        state = (pos, t)
        if state in visited:
            continue
        visited[state] = True

        for npos in env.neighbors(pos):
            nstate = (npos, t+1)
            if nstate not in visited and (npos, t+1) not in constraints:
                g = t + 1
                h = manhattan(npos, goal)
                heapq.heappush(open_list, (g+h, g, npos, path+[npos]))

    return None   # no path found


# ─────────────────────────────────────────────────────────────────────────────
# LP TASK ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

def lp_task_allocation(env: DisasterEnv) -> Dict[int, int]:
    """
    Linear Programming task allocation.

    Objective : Minimize Σ_i Σ_j c_ij * x_ij
                where c_ij = Manhattan distance from agent i to task j

    Constraints:
      - Each task assigned to exactly one agent: Σ_i x_ij = 1 ∀j
      - Each agent gets at most ceil(n_tasks/n_agents) tasks (fairness)
      - x_ij ∈ {0,1}  (relaxed to [0,1] for LP, then rounded)

    Returns: dict mapping agent_id → list of task_ids
    """
    agents = [a for a in env.agents]
    tasks  = [t for t in env.tasks if not t.completed]

    if not tasks:
        return {a.id: [] for a in agents}

    n_a = len(agents)
    n_t = len(tasks)

    # Cost matrix: c[i][j] = distance from agent i to task j
    cost = np.array([
        [manhattan(agents[i].position, tasks[j].position)
         for j in range(n_t)]
        for i in range(n_a)
    ], dtype=float)

    # Flatten for linprog: x has shape (n_a * n_t,)
    c_flat = cost.flatten()

    # Equality constraints: each task assigned to exactly one agent
    # Σ_i x_ij = 1 for each j
    A_eq = np.zeros((n_t, n_a * n_t))
    for j in range(n_t):
        for i in range(n_a):
            A_eq[j, i * n_t + j] = 1.0
    b_eq = np.ones(n_t)

    # Inequality constraints: fairness — max tasks per agent
    max_tasks = int(np.ceil(n_t / n_a)) + 1
    A_ub = np.zeros((n_a, n_a * n_t))
    for i in range(n_a):
        A_ub[i, i*n_t:(i+1)*n_t] = 1.0
    b_ub = np.full(n_a, max_tasks, dtype=float)

    bounds = [(0, 1)] * (n_a * n_t)

    result = linprog(c_flat, A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    # Assign tasks: for each task, pick agent with highest x_ij
    assignment: Dict[int, List[int]] = {a.id: [] for a in agents}
    if result.success:
        x = result.x.reshape(n_a, n_t)
        for j in range(n_t):
            best_agent = int(np.argmax(x[:, j]))
            assignment[agents[best_agent].id].append(tasks[j].id)
    else:
        # Fallback: greedy assignment
        assignment = greedy_task_allocation(env)

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY TASK ALLOCATION (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_task_allocation(env: DisasterEnv) -> Dict[int, List[int]]:
    """
    Greedy: assign each task to the nearest available agent.
    No fairness consideration — used as baseline comparison.
    """
    assignment: Dict[int, List[int]] = {a.id: [] for a in env.agents}
    tasks = [t for t in env.tasks if not t.completed]

    for task in tasks:
        dists = [manhattan(a.position, task.position) for a in env.agents]
        best  = int(np.argmin(dists))
        assignment[env.agents[best].id].append(task.id)

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# FE-MADDPG FAIRNESS REWARD ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def fe_maddpg_adjust_assignment(env: DisasterEnv,
                                 base_assignment: Dict[int, List[int]],
                                 n_iterations: int = 10) -> Dict[int, List[int]]:
    """
    Simulates the FE-MADDPG fairness adjustment step.

    Iteratively rebalances task loads using the fairness reward signal:
      r_t^i = (ε + |e_t^i / ē_t - 1|) / ē_t

    Agents with below-average performance (overloaded) shed tasks
    to agents with above-average performance (underloaded).

    This is a simplified simulation of the policy update loop in FE-MADDPG.
    """
    assignment = copy.deepcopy(base_assignment)

    for _ in range(n_iterations):
        # Compute current loads
        loads = {aid: len(tids) for aid, tids in assignment.items()}
        avg   = np.mean(list(loads.values())) if loads else 1.0

        # Performance: inverse of load (more tasks = worse perf)
        perfs = {aid: 1.0/(loads[aid]+env.epsilon) for aid in loads}
        e_bar = np.mean(list(perfs.values()))

        # Fairness reward per agent
        rewards = {}
        for aid, e_i in perfs.items():
            rewards[aid] = (env.epsilon + abs(e_i/e_bar - 1)) / e_bar

        # Find most overloaded and most underloaded
        most_loaded   = max(loads, key=loads.get)
        least_loaded  = min(loads, key=loads.get)

        if loads[most_loaded] - loads[least_loaded] <= 1:
            break   # balanced enough

        # Transfer one task from most loaded to least loaded
        if assignment[most_loaded]:
            task_id = assignment[most_loaded].pop()
            assignment[least_loaded].append(task_id)

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# CONFLICT-BASED SEARCH (CBS)
# ─────────────────────────────────────────────────────────────────────────────

def detect_conflicts(paths: Dict[int, List]) -> Optional[Tuple]:
    """
    Detect first conflict between any two agent paths.
    Returns (agent_i, agent_j, position, timestep) or None.
    """
    max_t = max((len(p) for p in paths.values()), default=0)

    for t in range(max_t):
        positions_at_t = {}
        for aid, path in paths.items():
            pos = path[t] if t < len(path) else path[-1]
            if pos in positions_at_t:
                return (positions_at_t[pos], aid, pos, t)
            positions_at_t[pos] = aid

        # Edge conflicts (swap)
        agents = list(paths.keys())
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                ai, aj = agents[i], agents[j]
                pi = paths[ai]
                pj = paths[aj]
                if t+1 < len(pi) and t+1 < len(pj):
                    if pi[t] == pj[t+1] and pj[t] == pi[t+1]:
                        return (ai, aj, pi[t], t)

    return None


def cbs(env: DisasterEnv,
        starts: Dict[int, Tuple],
        goals:  Dict[int, Tuple]) -> Dict[int, List]:
    """
    Conflict-Based Search (CBS) for multi-agent pathfinding.

    High level : constraint tree, resolve conflicts by adding constraints
    Low level  : A* with space-time constraints per agent

    Returns: dict mapping agent_id → collision-free path
    """
    # Initial paths (no constraints)
    init_paths = {}
    for aid in starts:
        path = astar(env, starts[aid], goals[aid])
        init_paths[aid] = path if path else [starts[aid]]

    root = CBSNode(
        cost=sum(len(p) for p in init_paths.values()),
        constraints=[],
        paths=init_paths
    )
    root.conflicts = []

    open_list = [root]
    heapq.heapify(open_list)

    best_cost = float('inf')
    best_paths = init_paths

    iterations = 0
    max_iterations = 500

    while open_list and iterations < max_iterations:
        iterations += 1
        node = heapq.heappop(open_list)

        conflict = detect_conflicts(node.paths)
        if conflict is None:
            # No conflicts — solution found
            if node.cost < best_cost:
                best_cost  = node.cost
                best_paths = node.paths
            break

        ai, aj, pos, t = conflict

        for (agent_id, constrained_pos, constrained_t) in [
            (ai, pos, t), (aj, pos, t)
        ]:
            new_constraints = node.constraints + [(agent_id, constrained_pos, constrained_t)]
            constraint_set  = set((c[1], c[2]) for c in new_constraints if c[0]==agent_id)

            new_path = astar(env, starts[agent_id], goals[agent_id],
                             constraints=constraint_set)
            if new_path is None:
                continue

            new_paths = copy.copy(node.paths)
            new_paths[agent_id] = new_path
            new_cost  = sum(len(p) for p in new_paths.values())

            child = CBSNode(
                cost=new_cost,
                constraints=new_constraints,
                paths=new_paths
            )
            heapq.heappush(open_list, child)

    return best_paths


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM ORCHESTRATORS
# ─────────────────────────────────────────────────────────────────────────────

def run_lp_cbs(env: DisasterEnv, use_fe_maddpg: bool = True) -> Dict:
    """
    Full LP + CBS pipeline (the paper's proposed method).

    1. LP task allocation
    2. (optional) FE-MADDPG fairness adjustment
    3. CBS pathfinding
    4. Simulate execution & collect metrics
    """
    t0 = time.perf_counter()

    # Step 1: LP allocation
    assignment = lp_task_allocation(env)

    # Step 2: FE-MADDPG adjustment
    if use_fe_maddpg:
        assignment = fe_maddpg_adjust_assignment(env, assignment)

    # Step 3: Build goal sequence per agent (ordered by proximity)
    all_paths: Dict[int, List] = {}
    total_steps = 0
    task_map = {t.id: t for t in env.tasks}

    for agent in env.agents:
        agent_tasks = [task_map[tid] for tid in assignment[agent.id]
                       if tid in task_map]
        # Order tasks by nearest neighbor
        ordered = _order_tasks_nn(agent.position, agent_tasks)

        full_path = [agent.position]
        current   = agent.position

        for task in ordered:
            # CBS between current pos and task (single agent sub-problem)
            segment = astar(env, current, task.position)
            if segment:
                full_path.extend(segment[1:])
                current = task.position

        all_paths[agent.id] = full_path
        total_steps = max(total_steps, len(full_path))

    # Step 4: CBS to resolve inter-agent conflicts
    # (run CBS on final goal only for efficiency)
    goals = {}
    for agent in env.agents:
        agent_tasks = [task_map[tid] for tid in assignment[agent.id]
                       if tid in task_map]
        if agent_tasks:
            ordered = _order_tasks_nn(agent.position, agent_tasks)
            goals[agent.id] = ordered[-1].position
        else:
            goals[agent.id] = agent.position

    starts = {a.id: a.position for a in env.agents}
    cbs_paths = cbs(env, starts, goals)

    # Merge: use CBS-resolved paths
    for aid in cbs_paths:
        if len(cbs_paths[aid]) > 1:
            all_paths[aid] = cbs_paths[aid]

    elapsed = time.perf_counter() - t0

    return _collect_metrics(env, assignment, all_paths, elapsed, 'LP_CBS_FE-MADDPG' if use_fe_maddpg else 'LP_CBS')


def run_greedy(env: DisasterEnv) -> Dict:
    """Greedy baseline: nearest task assignment + A* paths (no fairness)."""
    t0 = time.perf_counter()

    assignment = greedy_task_allocation(env)
    task_map   = {t.id: t for t in env.tasks}
    all_paths  = {}

    for agent in env.agents:
        agent_tasks = [task_map[tid] for tid in assignment[agent.id]
                       if tid in task_map]
        ordered   = _order_tasks_nn(agent.position, agent_tasks)
        full_path = [agent.position]
        current   = agent.position

        for task in ordered:
            segment = astar(env, current, task.position)
            if segment:
                full_path.extend(segment[1:])
                current = task.position

        all_paths[agent.id] = full_path

    elapsed = time.perf_counter() - t0
    return _collect_metrics(env, assignment, all_paths, elapsed, 'Greedy')


def run_cbs_only(env: DisasterEnv) -> Dict:
    """CBS pathfinding with greedy allocation (no LP fairness)."""
    t0 = time.perf_counter()

    assignment = greedy_task_allocation(env)
    task_map   = {t.id: t for t in env.tasks}

    goals  = {}
    for agent in env.agents:
        agent_tasks = [task_map[tid] for tid in assignment[agent.id]
                       if tid in task_map]
        if agent_tasks:
            ordered = _order_tasks_nn(agent.position, agent_tasks)
            goals[agent.id] = ordered[-1].position
        else:
            goals[agent.id] = agent.position

    starts    = {a.id: a.position for a in env.agents}
    cbs_paths = cbs(env, starts, goals)

    elapsed = time.perf_counter() - t0
    return _collect_metrics(env, assignment, cbs_paths, elapsed, 'CBS_only')


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _order_tasks_nn(start: Tuple, tasks: List[Task]) -> List[Task]:
    """Order tasks using nearest-neighbor heuristic."""
    if not tasks:
        return []
    remaining = list(tasks)
    ordered   = []
    current   = start
    while remaining:
        nearest = min(remaining, key=lambda t: manhattan(current, t.position))
        ordered.append(nearest)
        current = nearest.position
        remaining.remove(nearest)
    return ordered


def _collect_metrics(env: DisasterEnv,
                     assignment: Dict[int, List[int]],
                     paths: Dict[int, List],
                     elapsed: float,
                     algorithm: str) -> Dict:
    """Compute fairness and efficiency metrics."""
    loads        = [len(assignment[a.id]) for a in env.agents]
    path_lengths = [len(paths.get(a.id, [])) for a in env.agents]

    # Fairness metrics
    load_variance  = float(np.var(loads))
    load_std       = float(np.std(loads))
    gini           = _gini(loads)

    # Efficiency metrics
    total_path_len   = sum(path_lengths)
    max_makespan     = max(path_lengths) if path_lengths else 0
    avg_path_len     = float(np.mean(path_lengths))

    # Fairness rewards (simulated)
    perfs  = [1.0/(l + env.epsilon) for l in loads]
    e_bar  = np.mean(perfs) if perfs else 1.0
    f_rewards = [(env.epsilon + abs(e/e_bar - 1))/e_bar for e in perfs]

    return {
        'algorithm':       algorithm,
        'assignment':      assignment,
        'paths':           paths,
        'task_loads':      loads,
        'path_lengths':    path_lengths,
        'load_variance':   load_variance,
        'load_std':        load_std,
        'gini':            gini,
        'total_path_len':  total_path_len,
        'max_makespan':    max_makespan,
        'avg_path_len':    avg_path_len,
        'fairness_rewards':f_rewards,
        'solve_time_s':    elapsed,
        'n_agents':        env.n_agents,
        'n_tasks':         env.n_tasks,
    }


def _gini(values: List) -> float:
    """Gini coefficient — 0 = perfect equality, 1 = max inequality."""
    arr = np.array(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n   = len(arr)
    idx = np.arange(1, n+1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n+1)/n)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-EPISODE EXPERIMENT  (reproduces paper's Figure 3)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(algorithm: str = 'LP_CBS',
                   n_episodes: int = 100,
                   grid_size: int = 10,
                   n_agents: int = 5,
                   n_tasks: int = 10,
                   n_obstacles: int = 10,
                   base_seed: int = 42) -> Dict:
    """
    Run multiple episodes and collect average solve time per episode.
    Reproduces paper Fig 3: "Average Time Comparison Between Non-LP and LP Approaches"

    algorithm: 'LP_CBS' | 'LP_CBS_FE' | 'Greedy' | 'CBS_only'
    """
    solve_times = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        env  = DisasterEnv(grid_size=grid_size, n_agents=n_agents,
                           n_tasks=n_tasks, n_obstacles=n_obstacles, seed=seed)

        if algorithm == 'LP_CBS':
            res = run_lp_cbs(env, use_fe_maddpg=False)
        elif algorithm == 'LP_CBS_FE':
            res = run_lp_cbs(env, use_fe_maddpg=True)
        elif algorithm == 'Greedy':
            res = run_greedy(env)
        elif algorithm == 'CBS_only':
            res = run_cbs_only(env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        solve_times.append(res['solve_time_s'] * 1000)  # ms

    return {
        'algorithm':   algorithm,
        'solve_times': solve_times,
        'mean_time':   float(np.mean(solve_times)),
        'std_time':    float(np.std(solve_times)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_grid(env: DisasterEnv, paths: Dict[int, List] = None,
              title: str = "Grid World — Agents Navigating to Tasks",
              ax: plt.Axes = None):
    """Visualize grid world with agents, tasks, obstacles and paths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    gs = env.grid_size
    ax.set_xlim(-0.5, gs-0.5)
    ax.set_ylim(-0.5, gs-0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(gs))
    ax.set_yticks(range(gs))
    ax.grid(True, color='lightgray', linewidth=0.5)
    ax.set_facecolor('#f8f4f0')

    # Draw cells
    for r in range(gs):
        for c in range(gs):
            if env.grid[r, c] == CELL_OBSTACLE:
                rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, color='black', zorder=2)
                ax.add_patch(rect)
            elif env.grid[r, c] == CELL_FIRE:
                rect = plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                     color='#cc3300', alpha=0.7, zorder=2)
                ax.add_patch(rect)
                ax.text(c, r, '🔥', ha='center', va='center',
                        fontsize=10, zorder=3)

    # Draw paths
    if paths:
        for agent in env.agents:
            path = paths.get(agent.id, [])
            if len(path) > 1:
                xs = [p[1] for p in path]
                ys = [p[0] for p in path]
                ax.plot(xs, ys, color=agent.color,
                        linewidth=2, alpha=0.6, zorder=3)

    # Draw agents (triangles)
    for agent in env.agents:
        r, c = agent.position
        triangle = plt.Polygon(
            [[c, r+0.35], [c-0.3, r-0.3], [c+0.3, r-0.3]],
            color=agent.color, zorder=5
        )
        ax.add_patch(triangle)
        ax.text(c, r-0.45, f'A{agent.id}', ha='center', va='top',
                fontsize=7, color=agent.color, fontweight='bold')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Legend
    patches = [mpatches.Patch(color=a.color, label=f'Agent {a.id}')
               for a in env.agents]
    patches += [
        mpatches.Patch(color='black',   label='Obstacle'),
        mpatches.Patch(color='#cc3300', label='Fire Task'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=7,
              framealpha=0.9, ncol=2)

    return ax


def plot_task_allocation(results_before: Dict, results_after: Dict,
                         ax: plt.Axes = None):
    """Bar chart: task allocation before vs after FE-MADDPG (paper Fig 2)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    n_agents    = results_before['n_agents']
    agent_names = [f'Agent_{i+1}' for i in range(n_agents)]
    x = np.arange(n_agents)
    w = 0.35

    ax.bar(x - w/2, results_before['task_loads'], w,
           label='Before FE-MADDPG', color='#4472C4', alpha=0.9)
    ax.bar(x + w/2, results_after['task_loads'],  w,
           label='After FE-MADDPG',  color='#ED7D31', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.set_ylabel('Number of Tasks')
    ax.set_title('Comparison of Task Allocation\nBefore and After FE-MADDPG',
                 fontweight='bold')
    ax.legend()
    ax.set_ylim(0, max(max(results_before['task_loads']),
                        max(results_after['task_loads'])) + 2)
    ax.grid(axis='y', alpha=0.3)
    return ax


def plot_solve_times(lp_times: List[float], non_lp_times: List[float],
                     ax: plt.Axes = None):
    """Line chart comparing LP vs non-LP solve times (paper Fig 3)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    episodes = range(1, len(lp_times)+1)
    ax.plot(episodes, non_lp_times, color='#4472C4',
            linewidth=1.5, label='Non-LP Average Time', alpha=0.8)
    ax.plot(episodes, lp_times, color='#ED7D31',
            linewidth=1.5, label='LP Average Time', alpha=0.8)

    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Average Time to Solve Maze (ms)')
    ax.set_title('Average Time Comparison Between Non-LP and LP Approaches',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_fairness_comparison(results_list: List[Dict], ax: plt.Axes = None):
    """Radar / bar chart comparing fairness metrics across algorithms."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    algorithms = [r['algorithm'] for r in results_list]
    ginis      = [r['gini']          for r in results_list]
    variances  = [r['load_variance'] for r in results_list]

    x = np.arange(len(algorithms))
    w = 0.35
    ax.bar(x - w/2, ginis,     w, label='Gini Coefficient (↓ = fairer)',
           color='#4472C4', alpha=0.9)
    ax.bar(x + w/2, variances, w, label='Load Variance (↓ = fairer)',
           color='#ED7D31', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=15, ha='right')
    ax.set_title('Fairness Comparison Across Algorithms', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: RUN ONE SCENARIO & RETURN ALL RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def compare_algorithms(seed: int = 42,
                       grid_size: int = 10,
                       n_agents: int = 5,
                       n_tasks: int = 10,
                       n_obstacles: int = 10) -> Dict[str, Dict]:
    """
    Run all algorithms on the SAME environment (same seed) and return results.
    Used for fair side-by-side comparison.
    """
    results = {}

    for algo in ['Greedy', 'CBS_only', 'LP_CBS', 'LP_CBS_FE']:
        env = DisasterEnv(grid_size=grid_size, n_agents=n_agents,
                          n_tasks=n_tasks, n_obstacles=n_obstacles, seed=seed)
        if algo == 'Greedy':
            res = run_greedy(env)
        elif algo == 'CBS_only':
            res = run_cbs_only(env)
        elif algo == 'LP_CBS':
            res = run_lp_cbs(env, use_fe_maddpg=False)
        elif algo == 'LP_CBS_FE':
            res = run_lp_cbs(env, use_fe_maddpg=True)
        results[algo] = res

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Disaster Response Multi-Agent Simulation")
    print("=" * 60)

    SEED = 42
    results = compare_algorithms(seed=SEED)

    print(f"\n{'Algorithm':<20} {'Gini':>8} {'Variance':>10} {'Makespan':>10} {'Time(ms)':>10}")
    print("-" * 60)
    for algo, r in results.items():
        print(f"{algo:<20} {r['gini']:>8.4f} {r['load_variance']:>10.4f} "
              f"{r['max_makespan']:>10d} {r['solve_time_s']*1000:>10.2f}")

    print("\nDone. Import this module in the Jupyter notebook for interactive use.")
