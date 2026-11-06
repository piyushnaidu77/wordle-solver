"""Bilinear REINFORCE policy. Every allowed guess is a legal action."""

from __future__ import annotations

import numpy as np

from solver.engine import World

BELIEF = 182  # 5*26 position mask + min counts + max counts
GUESS = 130


def play(world: World, choose, target: int) -> tuple[list[str], bool]:
    remaining = np.arange(world.n_answers, dtype=np.int32)
    secret = world.answers[target]
    guesses: list[str] = []
    for _ in range(6):
        guess = choose(remaining)
        word = world.legal[guess]
        guesses.append(word)
        if word == secret:
            return guesses, True
        pattern = int(world.patterns[guess, target])
        remaining = remaining[world.patterns[guess, remaining] == pattern]
        if remaining.size == 0:
            break
    return guesses, False


def summarize(solved_turns: list[int | None]) -> dict:
    histogram = [0] * 6
    fails = 0
    cost = 0
    solved_total = 0
    solved_count = 0
    for outcome in solved_turns:
        if outcome is None:
            fails += 1
            cost += 7
        else:
            histogram[outcome - 1] += 1
            cost += outcome
            solved_total += outcome
            solved_count += 1
    games = len(solved_turns)
    return {
        "games": games,
        "fails": fails,
        "histogram": {str(i + 1): histogram[i] for i in range(6)},
        "mean_cost": cost / games,
        "mean_guesses_when_solved": (solved_total / solved_count) if solved_count else None,
    }


class ReinforceAgent:
    def __init__(self, world: World, seed: int = 0) -> None:
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0.0, 0.01, size=(GUESS, BELIEF)).astype(np.float32)
        # Starts focused on words still consistent with the clues. This weight is learned.
        self.possible_weight = np.float32(5.0)
        self.baseline = np.full(6, -4.0, dtype=np.float64)
        self.learning_rate = 0.05

    def act(self, remaining: np.ndarray, greedy: bool) -> tuple[int, dict]:
        belief = self.world.belief(remaining)
        possible = np.zeros(self.world.n_guesses, dtype=np.float64)
        possible[remaining] = 1.0
        hidden = self.weights @ belief
        scores = self.world.guess_onehot @ hidden
        scores += self.possible_weight * possible
        probs = self._distribution(scores)
        if greedy:
            local = int(np.argmax(scores))
        else:
            local = int(self.rng.choice(self.world.n_guesses, p=probs))
        info = {
            "belief": belief,
            "possible": possible,
            "probs": probs,
            "local": local,
        }
        return local, info

    def _distribution(self, scores: np.ndarray) -> np.ndarray:
        shifted = scores - np.max(scores)
        weights = np.exp(shifted.astype(np.float64))
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            return np.full(scores.shape[0], 1.0 / scores.shape[0])
        return weights / total

    def train(self, episodes: int) -> list[float]:
        costs: list[float] = []
        window: list[float] = []
        for episode in range(episodes):
            target = int(self.rng.integers(0, self.world.n_answers))
            steps, solved_turn = self._episode(target, greedy=False)
            total = -solved_turn if solved_turn else -7
            returns = self._returns(len(steps), solved_turn is not None)
            for turn, info in enumerate(steps):
                advantage = returns[turn] - self.baseline[turn]
                self.baseline[turn] += 0.02 * (returns[turn] - self.baseline[turn])
                self._update(info, advantage)
            costs.append(-total)
            window.append(-total)
            if len(window) == 1000:
                print(f"train {episode + 1}/{episodes} mean cost {sum(window) / len(window):.3f}", flush=True)
                window.clear()
        return costs

    def _returns(self, turns: int, solved: bool) -> list[float]:
        rewards = [-1.0] * turns
        if not solved:
            rewards[-1] += -7.0 - sum(rewards)
        returns = [0.0] * turns
        running = 0.0
        for index in range(turns - 1, -1, -1):
            running += rewards[index]
            returns[index] = running
        return returns

    def _update(self, info: dict, advantage: float) -> None:
        onehot = self.world.guess_onehot
        expected = info["probs"] @ onehot
        chosen = onehot[info["local"]]
        gradient = np.outer(chosen - expected, info["belief"]).astype(np.float32)
        norm = float(np.linalg.norm(gradient))
        if norm > 5.0:
            gradient *= np.float32(5.0 / norm)
        self.weights += np.float32(self.learning_rate * advantage) * gradient
        possible = info["possible"]
        possible_grad = possible[info["local"]] - float(info["probs"] @ possible)
        self.possible_weight += np.float32(self.learning_rate * advantage * possible_grad)

    def _episode(self, target: int, greedy: bool) -> tuple[list[dict], int | None]:
        remaining = np.arange(self.world.n_answers, dtype=np.int32)
        secret = self.world.answers[target]
        steps: list[dict] = []
        for _ in range(6):
            guess, info = self.act(remaining, greedy=greedy)
            steps.append(info)
            if self.world.legal[guess] == secret:
                return steps, len(steps)
            pattern = int(self.world.patterns[guess, target])
            remaining = remaining[self.world.patterns[guess, remaining] == pattern]
            if remaining.size == 0:
                break
        return steps, None

    def solve_all(self) -> dict[str, list[str]]:
        lines: dict[str, list[str]] = {}
        turns: list[int | None] = []
        for target, answer in enumerate(self.world.answers):
            guesses, solved = play(self.world, lambda remaining: self.act(remaining, greedy=True)[0], target)
            lines[answer] = guesses
            turns.append(len(guesses) if solved else None)
        self.last_report = summarize(turns)
        return lines
