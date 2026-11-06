"""Train the REINFORCE policy and write its guess list for every answer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solver.engine import World
from solver.reinforce import ReinforceAgent

EPISODES = 20000
RESULTS = Path(__file__).resolve().parent / "results.json"
SOLUTIONS = Path(__file__).resolve().parents[1] / "data" / "solutions.json"


def main() -> None:
    print("loading words and patterns", flush=True)
    world = World()
    print(f"answers {world.n_answers} allowed {world.n_guesses}", flush=True)
    agent = ReinforceAgent(world)
    print(f"training reinforce for {EPISODES} episodes", flush=True)
    costs = agent.train(EPISODES)
    early = float(np.mean(costs[:1000]))
    late = float(np.mean(costs[-1000:]))
    print(f"train cost first 1000 {early:.3f} last 1000 {late:.3f}", flush=True)
    print("writing guess lines", flush=True)
    lines = agent.solve_all()
    SOLUTIONS.parent.mkdir(exist_ok=True)
    SOLUTIONS.write_text(json.dumps(lines, separators=(",", ":")), encoding="utf-8")
    report = agent.last_report
    report["episodes"] = EPISODES
    report["possible_weight"] = float(agent.possible_weight)
    report["train_cost_first_1000"] = early
    report["train_cost_last_1000"] = late
    payload = {
        "answers": world.n_answers,
        "allowed_guesses": world.n_guesses,
        "reward": "Solving in n guesses returns -n. Failing after 6 returns -7.",
        "reinforce": report,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"mean cost {report['mean_cost']:.4f} fails {report['fails']} wrote {SOLUTIONS}",
        flush=True,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
