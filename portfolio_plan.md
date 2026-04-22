# ARIA Research Portfolio Plan
## Fair & Efficient Multi-Agent Disaster Response — Jack Pham

---

## WHAT YOU HAVE (THE ASSET STACK)

| Asset | File | Purpose |
|---|---|---|
| Original paper | `Jack_Pham_.pdf` | Academic credibility anchor |
| Research poster | `poster.png` | Visual proof of conference-level work |
| Simulation engine | `disaster_response_sim.py` | Runnable Python, shows depth |
| Jupyter notebook | `disaster_response.ipynb` | Explainability + reproducibility |
| Live sim app | `simulation.html` | The showstopper demo |

---

## PART 1 — GITHUB REPOSITORY

### Repo Name: `aria-disaster-response`
### Structure:
```
aria-disaster-response/
├── README.md                  ← THE most important file
├── simulation.html            ← Self-contained live demo
├── disaster_response_sim.py   ← Core engine
├── disaster_response.ipynb    ← Research notebook
├── assets/
│   ├── poster.png
│   ├── demo.gif               ← Screen record the simulation (see below)
│   └── fig1_agents.png        ← Generated figures from notebook
├── docs/                      ← GitHub Pages site (optional)
└── paper/
    └── Jack_Pham_REU_Paper.pdf
```

### README.md must include:
1. **Header badge strip**: `NSF REU #2150390` | `Python 3.10` | `Live Demo` link
2. **One GIF of the simulation** (most viewed part of any ML/robotics repo)
3. **The core equation** rendered in LaTeX via GitHub: `$r_t^i = \frac{\varepsilon + |e_t^i/\bar{e}_t - 1|}{\bar{e}_t}$`
4. **Algorithm comparison table** with actual numbers from your runs
5. **Quick start** — `python disaster_response_sim.py` working in <60 seconds
6. **Citation block** for your paper

### GitHub Pages (free):
- Enable GitHub Pages from `/docs` or `main` branch
- Drop `simulation.html` → becomes live at `https://jackpham.github.io/aria-disaster-response`
- This is your demo URL you put everywhere

---

## PART 2 — LAB/RESEARCH LOG (Notion or Obsidian)

Create a public Notion page titled:
**"NSF REU Summer Research Log — ARIA Project"**

### Weekly Log Structure:
```
Week 1: Problem framing, literature review (Huang 2021, Liu 2022)
Week 2: MDP formulation, grid world setup
Week 3: A* + CBS implementation
Week 4: LP task allocation
Week 5: FE-MADDPG fairness reward integration
Week 6: Experiments + Figure generation
Week 7: Poster design, paper writing
Week 8: Presentation, future work scoping
```

### Why this matters:
Recruiters and grad school reviewers LOVE seeing the process, not just results.
A timestamped research log proves you did the work and shows intellectual growth.

---

## PART 3 — PERSONAL WEBSITE

### Recommended Stack: **Astro** or plain HTML (fast, free, no framework overhead)
### Host: **Vercel** or **GitHub Pages** (both free)

### Page structure for ARIA project:
```
/projects/aria-disaster-response
  ↳ Hero: Title + simulation embed (iframe your simulation.html)
  ↳ Problem: The disaster response challenge + NSF badge
  ↳ Methods: Equations rendered with KaTeX, algorithm comparison table
  ↳ Results: Your 3 figures (Fig 1, 2, 3 from notebook)
  ↳ Demo: Embedded simulation OR link to GitHub Pages
  ↳ Paper: PDF download link
  ↳ Code: GitHub link
```

### Embed the simulation directly on your portfolio:
```html
<iframe
  src="https://yourname.github.io/aria-disaster-response/simulation.html"
  width="100%" height="600px" frameborder="0"
  style="border-radius:8px; box-shadow:0 4px 32px rgba(0,0,0,0.3)">
</iframe>
```

---

## PART 4 — VIDEO (YES, MAKE ONE)

### Why: LinkedIn posts with video get 5× more engagement. Grad school apps love demos.

### What to record (2-3 minutes max):
1. **[0:00-0:20]** Open simulation → brief title card with your name + NSF logo
2. **[0:20-0:50]** Show GREEDY algorithm running → point out unequal task loads
3. **[0:50-1:30]** Switch to LP+FE-MADDPG → show robots collaborating (the purple beams)
4. **[1:30-2:00]** Show the episode growth chart + metrics panel improving
5. **[2:00-2:30]** Quick cut to Jupyter notebook → show the equation + Fig 3

### Tools (free):
- **OBS Studio** → screen record the simulation
- **DaVinci Resolve** (free) → add title cards, cut
- Keep background music subtle (lofi, no copyright issues)

### Upload to:
- YouTube (unlisted or public) → embed link everywhere
- LinkedIn post → "Summer research I did at UT Austin..."
- Twitter/X thread → good for ML/robotics community

---

## PART 5 — GAZEBO / ROS QUESTION

### Honest assessment: Is it worth it?

**Short answer: Not for the portfolio right now — but here's the roadmap.**

| Option | Effort | Portfolio Value | Recommendation |
|---|---|---|---|
| Current HTML sim | ✅ Done | High (visual, interactive) | Ship this NOW |
| ROS2 + Gazebo port | 4-6 weeks | Very High (industry signal) | Do it in Phase 2 |
| ROS2 alone (no Gazebo) | 2-3 weeks | High | Good middle ground |

### Why the current sim is ENOUGH for now:
- Your paper + notebook + interactive demo already exceeds 90% of undergrad portfolios
- Employers/grad schools care that you understand the *theory* — the HTML sim proves that
- Gazebo without a physical robot or meaningful sensor sim is often just overhead

### When to do Gazebo (Phase 2):
- After you get comfortable with ROS2 basics (nav2, tf2)
- When you have a specific robot to simulate (TurtleBot, Jackal, etc.)
- If you're applying to robotics-specific jobs/labs (DARPA, JPL, robotics startups)

### ROS2 Quick Win (2 weeks, big signal):
Instead of full Gazebo, port just the CBS pathfinder + task allocator as a ROS2 package:
```
ros2 run aria_disaster_response allocator_node
ros2 run aria_disaster_response cbs_planner_node
ros2 topic echo /agent_assignments
```
This shows ROS knowledge without the full Gazebo complexity.

---

## PART 6 — TRAINING STRATEGY

### Option A: Train on Your Laptop
**Best for:** Quick experiments, development, debugging

Specs needed (you likely have this):
- Python 3.10+, NumPy, SciPy — already works
- For actual RL training (FE-MADDPG neural nets): PyTorch CPU

Timeline for laptop training:
```
Grid 10×10, 5 agents, 100 episodes  → ~2-5 min (CPU, fine)
Grid 20×20, 10 agents, 500 episodes → ~20-40 min (CPU, manageable)
Full RL (actor-critic nets) 10k eps → ~4-8 hours (CPU, overnight)
```
Recommendation: **Use laptop for all experiments except full RL training.**

### Option B: School HPC (TACC or your school's cluster)
**Best for:** Full FE-MADDPG with neural network policy, large-scale scalability tests

**Request access if:**
- You want to train real neural network policies (actor-critic, 10k+ episodes)
- You want to run parallel seeds (50 seeds × 1000 episodes)
- You want GPU-accelerated PyTorch

### TACC specifically (since Topcu's lab uses it):
You already have the connection through Dr. Topcu / NSF REU.
Email Dr. Thorpe: "I'm expanding the ARIA project — could I request an allocation on TACC Frontera/Lonestar6?"

HPC job script template:
```bash
#!/bin/bash
#SBATCH -J aria_training
#SBATCH -N 1 -n 16
#SBATCH -t 4:00:00
#SBATCH -p normal
module load python3
python disaster_response_sim.py --episodes 5000 --seed 42 --algo lp_fe --save results/
```

**Recommendation: Start on laptop. When you're ready for the full RL implementation, use TACC.**

---

## PART 7 — FUTURE WORK IMPLEMENTATION PRIORITY

From your paper's Future Work section, here's what to tackle first:

### Priority 1 (1-2 weeks): Real-time Seed Variation
- Run 500 episodes with different seeds, plot convergence
- This is the "episode growth" curve that impresses reviewers most
- Already half-built in your Jupyter notebook

### Priority 2 (2-4 weeks): Full FE-MADDPG with Neural Nets
- Replace the LP approximation with actual PyTorch actor-critic networks
- Shows you can do real deep RL, not just LP
- Would make the paper publishable at a stronger venue

### Priority 3 (4-6 weeks): ROS2 Package
- Wrap allocator + CBS as ROS2 nodes
- Add TurtleBot3 sim in Gazebo (use pre-built models)
- Record video of actual robot sim → massive portfolio upgrade

### Priority 4 (ongoing): Apply to More Scenarios
- Warehouse: n robots, m packages, fairness in delivery time
- Agriculture: multi-drone crop monitoring
- These are direct resume bullets: "Extended ARIA framework to warehouse logistics"

---

## IMMEDIATE ACTION CHECKLIST

```
This week:
  [ ] Create GitHub repo: aria-disaster-response
  [ ] Upload all 4 files (sim.py, .ipynb, .html, paper)
  [ ] Write README with demo GIF (record screen + use ezgif.com to convert)
  [ ] Enable GitHub Pages → share the live demo URL

Next 2 weeks:
  [ ] Add ARIA project to your personal website / portfolio
  [ ] Post on LinkedIn with video clip (tag UT Austin, NSF, Dr. Thorpe)
  [ ] Run 100-episode experiment in notebook → save the Figure 3 chart
  [ ] Email Dr. Thorpe: "I've open-sourced the ARIA project, here's the repo"

Month 2:
  [ ] Write a blog post / devlog about what you learned (Medium or GitHub blog)
  [ ] Start ROS2 basics (ros.org tutorials, free)
  [ ] Begin PyTorch implementation of actual actor-critic networks

For grad school apps:
  [ ] Reference this project in your Statement of Purpose
  [ ] List GitHub repo URL in your CV under Research Experience
  [ ] Request letter of rec from Dr. Thorpe + Dr. Topcu (reference THIS project)
```

---

## SUMMARY

Your summer research is genuinely impressive for an undergrad. The combination of:
- LP + RL (FE-MADDPG) + classical search (CBS)
- NSF funding acknowledgment
- UT Austin advisors
- Now: interactive simulation + reproducible notebook

...puts you well above average for both grad school applications and robotics/AI internships.

**The simulation is your hook. The paper is your credibility. The notebook is your depth.**
Ship it, share it, iterate.
