# Fair and Efficient Task Allocation and Path Planning for Multi-Agent Systems in Disaster Response Using Linear Programming

Back in the summer of 2024, I had the opportunity to participate in an NSF REU (Research Experience for Undergraduates) at the University of Texas at Austin, where I got to work alongside the Texas Advanced Computing Center (TACC) and Dr. Adam Thorpe on this project. It was honestly one of the coolest experiences I've had — getting to do real research as an undergrad and actually see it through to a paper and poster presentation.

---

## ⚠️ New approach — check this out first

After the REU I kept working on this problem and built a completely different version of the system. Instead of a single tier of robots, it uses **aerial drones that scout terrain and hand off to ground robots that perform rescues**, with fire trucks added for suppression. The algorithms are also upgraded to MAPPO and QMIX.

👉 **[drone-ugv-disaster-response](https://github.com/YOUR-USERNAME/drone-ugv-disaster-response)** — the follow-up project

👉 **[Live 3D simulation](https://YOUR-USERNAME.github.io/drone-ugv-disaster-response/sim3d_v3.html)** — runs in browser, no install

If you're here for the portfolio or just curious what this research became, that repo is the more polished version. This one below is the original REU work.

---

## What's the goal here?

The core idea is: when you send a team of robots into a disaster zone (think search and rescue, fire suppression, debris clearing), how do you assign tasks *fairly* and plan their paths *efficiently* so they don't crash into each other and no single robot gets overloaded?

We tackled this by combining three things:
- **Linear Programming (LP)** to allocate tasks in a way that balances workload across all robots
- **FE-MADDPG** (Fairness-Enhanced Multi-Agent Deep Deterministic Policy Gradient) to bake fairness directly into the reward signal each robot learns from
- **CBS** (Conflict-Based Search) to find collision-free paths for all robots simultaneously

The fairness reward function at the heart of it:

$$r_t^i = \frac{\varepsilon + \left|e_t^i / \bar{e}_t - 1\right|}{\bar{e}_t}$$

Where $e_t^i$ is an agent's performance relative to the average $\bar{e}_t$ — basically nudging agents toward equal contribution over time.

---

## Try the simulator

I built an interactive simulation that lets you watch the robots collaborate in real time — different robot roles (Scout, Medic, Engineer, Carrier, FireBot), different mission types, live fairness metrics, and the ability to switch between algorithms on the fly.

👉 **[Launch the 2D hex simulation](https://YOUR-USERNAME.github.io/area-disaster-response/simulation_v3.html)**

👉 **[View research data visualizations](https://YOUR-USERNAME.github.io/area-disaster-response/visualizations.html)**

Or clone the repo and open either HTML file directly in your browser — no setup needed.

---

## A little personal note

Even though this project lives in the world of reinforcement learning and multi-agent systems, since 2024 I've actually shifted my focus toward pure electrical engineering — specifically integrated circuits and signals. That's where my heart is these days, and that's the path I'm committed to going forward.

That said, I'm glad this work exists. The problem of fairness in autonomous systems is genuinely important, and I think there's a lot of promise in applying these ideas to real-world robotics down the road. The follow-up project above is my attempt at pushing it further — smarter coordination, 3D terrain, dynamic fire, and a two-tier drone/ground robot split.

---

## Thank you

To everyone I met through the REU program — Dr. Adam Thorpe, Dr. Ufuk Topcu, the folks at TACC, and my fellow REU cohort — thank you. That summer shaped how I think about research and problem-solving in ways I still carry with me.

This work was supported by the NSF REU Site: CI Research for Social Change, Award #2150390.

---

## Files

| File | Description |
|------|-------------|
| `simulation_v3.html` | Interactive 2D hex simulation (open in browser) |
| `visualizations.html` | Research data visualizations — all figures with charts |
| `disaster_response_sim.py` | Core simulation engine (Python) |
| `disaster_response.ipynb` | Original research notebook |
| `paper/` | Original REU paper (PDF) |
