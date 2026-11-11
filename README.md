# Fair and Efficient Task Allocation and Path Planning for Multi-Agent Systems in Disaster Response Using Linear Programming

My first research project: a team of robots in a disaster zone that splits up the work fairly and plans paths that never collide. Built during the 2024 NSF REU at the University of Texas at Austin with the Texas Advanced Computing Center (TACC). [Live simulation](https://jackpham-rgb.github.io/area-disaster-response/simulation_v2.html)

## Goal

Send a team of robots into a disaster zone to search, carry, clear debris and fight fires. Two things have to go right:

- **Fair split.** No single robot should end up doing most of the work while others sit idle.
- **Safe paths.** Robots share the same narrow space, so their routes must never put two of them in the same place at the same time.

## How it works

I learned that each of these problems already has a name and a standard tool, so I combined three of them.

- **Linear programming (LP).** Assigning tasks to robots can be written as "minimize total cost, subject to every task being covered and no robot taking too much." Problems in that form are called linear programs, and solvers handle them quickly. That is how tasks get allocated.
- **FE-MADDPG.** Reinforcement learning trains agents by reward. MADDPG is a version for many agents learning together. The FE stands for fairness-enhanced: each robot's reward includes a penalty when its workload drifts from the team average.
- **CBS (Conflict-Based Search).** Plan each robot's path on its own, check whether any two collide, and if they do, add a rule forbidding that spot at that time and replan only the robots involved. Repeat until there are no collisions.

The fairness reward:

$$r_t^i = \frac{\varepsilon + \left|e_t^i / \bar{e}_t - 1\right|}{\bar{e}_t}$$

Here $e_t^i$ is one robot's performance and $\bar{e}_t$ is the team average. The further a robot is from the average, the bigger the penalty, which nudges the team toward equal contribution.

## Try it

- **[Simulation v2](https://jackpham-rgb.github.io/area-disaster-response/simulation_v2.html)**: hexagonal terrain with five robot roles (Scout, Medic, Engineer, Carrier, FireBot), mission types, live fairness metrics, and a switch between algorithms while it runs.
- **[Simulation v1](https://jackpham-rgb.github.io/area-disaster-response/simulation.html)**: the first, square-grid version.
- **[Data visualizations](https://jackpham-rgb.github.io/area-disaster-response/visualizations.html)**: the five figures from the research.

Each page also opens straight from a cloned copy of the repo in any browser.

On the simulation pages, the Gini index measures how unevenly work is spread. 0 means perfectly equal. I watch it drop when switching from the greedy baseline to LP + FE-MADDPG.

## Files

| File | What it is |
|---|---|
| `simulation_v2.html` | Interactive hex-grid simulation |
| `simulation.html` | Interactive square-grid simulation (first version) |
| `visualizations.html` | Research figures with charts |
| `disaster_response_sim.py` | The simulation engine in Python |
| `disaster_response.ipynb` | Notebook with the theory, figures and experiments |
| `REU paper.pdf` | The REU report |

## Where it went next

[drone-ugv-disaster-response](https://github.com/jackpham-rgb/drone-ugv-disaster-response) builds on this. It splits the team into drones that scout and ground robots that rescue, and trains the policies with deep reinforcement learning.

## Thank you

To everyone I met through the REU program: Dr. Adam Thorpe, Dr. Ufuk Topcu, Dr. Rosalia Gomez, my friends at TACC and my REU cohort. That summer shaped how I think about research and problem solving.

This work was supported by the NSF REU Site: CI Research for Social Change, Award #2150390.
