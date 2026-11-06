"""Wordle simulator: legal guesses, green-then-yellow patterns, belief sets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORDS = ROOT / "words"
CACHE = Path(__file__).resolve().parent / "cache"
ANSWERS_PATH = WORDS / "wordle-answers-alphabetical.txt"
GUESSES_PATH = WORDS / "wordle-allowed-guesses.txt"
ALL_GREEN = 3**5 - 1  # 242, every position correct


def parse_list(path: Path) -> list[str]:
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip().lower()
        if len(word) == 5 and word.isalpha():
            words.append(word)
    return words


def load_lists() -> tuple[list[str], list[str]]:
    answers = parse_list(ANSWERS_PATH)
    seen = set(answers)
    legal = list(answers)
    for word in parse_list(GUESSES_PATH):
        if word not in seen:
            legal.append(word)
            seen.add(word)
    if len(answers) != len(set(answers)):
        raise RuntimeError("duplicate answers")
    return answers, legal


def chars(words: list[str]) -> np.ndarray:
    out = np.empty((len(words), 5), dtype=np.int8)
    for i, word in enumerate(words):
        out[i] = [ord(ch) - 97 for ch in word]
    return out


def pattern_scalar(guess: np.ndarray, target: np.ndarray) -> int:
    """Green first, then yellows consume what is left. Same rule as js/game.js."""
    result = [0, 0, 0, 0, 0]
    counts = [0] * 26
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 2
        else:
            counts[target[i]] += 1
    for i in range(5):
        if result[i] == 2:
            continue
        if counts[guess[i]] > 0:
            result[i] = 1
            counts[guess[i]] -= 1
    code = 0
    for value in result:
        code = code * 3 + value
    return code


def patterns_for_guess(guess: np.ndarray, answers: np.ndarray) -> np.ndarray:
    count = answers.shape[0]
    green = answers == guess
    result = np.zeros((count, 5), dtype=np.int16)
    result[green] = 2
    counts = np.zeros((count, 26), dtype=np.int16)
    rows = np.arange(count)
    for i in range(5):
        np.add.at(counts, (rows, answers[:, i]), 1)
        counts[rows, guess[i]] -= green[:, i]
    for i in range(5):
        letter = int(guess[i])
        yellow = (~green[:, i]) & (counts[:, letter] > 0)
        result[yellow, i] = 1
        counts[yellow, letter] -= 1
    code = result[:, 0]
    for i in range(1, 5):
        code = code * 3 + result[:, i]
    return code.astype(np.uint8)


def check_patterns() -> None:
    samples = {
        ("papal", "apple"): [1, 1, 2, 0, 1],
        ("puppy", "apple"): [1, 0, 2, 0, 0],
        ("babes", "abbey"): [1, 1, 2, 2, 0],
        ("apple", "apple"): [2, 2, 2, 2, 2],
    }
    for (guess, target), marks in samples.items():
        g = np.array([ord(ch) - 97 for ch in guess], dtype=np.int8)
        t = np.array([ord(ch) - 97 for ch in target], dtype=np.int8)
        code = 0
        for mark in marks:
            code = code * 3 + mark
        got = pattern_scalar(g, t)
        if got != code:
            raise RuntimeError(f"{guess}/{target}: {got} != {code}")
        batch = patterns_for_guess(g, t[None, :])
        if int(batch[0]) != code:
            raise RuntimeError(f"vector {guess}/{target}: {int(batch[0])} != {code}")


def pattern_matrix(guess_chars: np.ndarray, answer_chars: np.ndarray) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "patterns.npy"
    meta_path = CACHE / "patterns.json"
    meta = {
        "guesses": int(guess_chars.shape[0]),
        "answers": int(answer_chars.shape[0]),
    }
    if path.exists() and meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved == meta:
            return np.load(path)
    total = guess_chars.shape[0]
    out = np.empty((total, answer_chars.shape[0]), dtype=np.uint8)
    for index in range(total):
        out[index] = patterns_for_guess(guess_chars[index], answer_chars)
        if index % 2000 == 0:
            print(f"patterns {index}/{total}", flush=True)
    np.save(path, out)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"patterns {total}/{total}", flush=True)
    return out


class World:
    def __init__(self) -> None:
        check_patterns()
        self.answers, self.legal = load_lists()
        self.answer_chars = chars(self.answers)
        self.guess_chars = chars(self.legal)
        self.patterns = pattern_matrix(self.guess_chars, self.answer_chars)
        self.n_answers = len(self.answers)
        self.n_guesses = len(self.legal)
        # Answers are the first legal words, so answer i is guess i.
        self.answer_to_guess = np.arange(self.n_answers, dtype=np.int32)
        self.guess_onehot = np.zeros((self.n_guesses, 130), dtype=np.float32)
        rows = np.arange(self.n_guesses)
        for i in range(5):
            self.guess_onehot[rows, i * 26 + self.guess_chars[:, i]] = 1.0
        rng = np.random.default_rng(0)
        for _ in range(30):
            guess = int(rng.integers(0, self.n_guesses))
            answer = int(rng.integers(0, self.n_answers))
            got = int(self.patterns[guess, answer])
            expected = pattern_scalar(self.guess_chars[guess], self.answer_chars[answer])
            if got != expected:
                raise RuntimeError(f"pattern mismatch {self.legal[guess]} {self.answers[answer]}")

    def belief(self, remaining: np.ndarray) -> np.ndarray:
        words = self.answer_chars[remaining]
        pos = np.zeros(130, dtype=np.float32)
        for i in range(5):
            pos[i * 26 + words[:, i]] = 1.0
        counts = np.zeros((words.shape[0], 26), dtype=np.int16)
        rows = np.arange(words.shape[0])
        for i in range(5):
            np.add.at(counts, (rows, words[:, i]), 1)
        count_min = counts.min(axis=0).astype(np.float32)
        count_max = counts.max(axis=0).astype(np.float32)
        return np.concatenate((pos, count_min, count_max))
