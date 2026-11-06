const ROWS = 6;
const COLS = 5;
const FLIP_MS = 500;
const STAGGER_MS = 280;
const RANK = { absent: 1, present: 2, correct: 3 };
const KEY_ROWS = ["qwertyuiop".split(""), "asdfghjkl".split(""), "zxcvbnm".split("")];

function scoreGuess(guess, target) {
  const result = Array(COLS).fill("absent");
  const counts = {};
  for (const ch of target) counts[ch] = (counts[ch] || 0) + 1;
  for (let i = 0; i < COLS; i += 1) {
    if (guess[i] === target[i]) {
      result[i] = "correct";
      counts[guess[i]] -= 1;
    }
  }
  for (let i = 0; i < COLS; i += 1) {
    if (result[i] === "correct") continue;
    if ((counts[guess[i]] || 0) > 0) {
      result[i] = "present";
      counts[guess[i]] -= 1;
    }
  }
  return result;
}

const $ = (id) => document.getElementById(id);

const state = {
  solutions: {},
  answers: [],
  shown: [],
  selected: "",
  run: 0,
  keys: {},
};

buildBoard();
buildKeyboard();
bind();
boot();

async function boot() {
  try {
    const response = await fetch("data/solutions.json");
    if (!response.ok) throw new Error("missing solutions");
    state.solutions = await response.json();
    state.answers = Object.keys(state.solutions).sort();
    if (!state.answers.length) throw new Error("empty solutions");
  } catch {
    $("loading").textContent = "Couldn't load the solver lines. Start the server from the project folder and reload.";
    return;
  }
  document.body.classList.add("ready");
  renderList("");
  $("filter").focus();
}

function buildBoard() {
  const board = $("board");
  for (let row = 0; row < ROWS; row += 1) {
    const rowEl = document.createElement("div");
    rowEl.className = "row";
    rowEl.setAttribute("role", "row");
    for (let col = 0; col < COLS; col += 1) {
      const tile = document.createElement("div");
      tile.className = "tile";
      tile.dataset.state = "empty";
      tile.setAttribute("role", "gridcell");
      tile.setAttribute("aria-label", "empty");
      rowEl.appendChild(tile);
    }
    board.appendChild(rowEl);
  }
}

function buildKeyboard() {
  const keyboard = $("keyboard");
  for (const row of KEY_ROWS) {
    const rowEl = document.createElement("div");
    rowEl.className = "kb-row";
    for (const key of row) {
      const button = document.createElement("div");
      button.className = "key";
      button.textContent = key;
      state.keys[key] = button;
      rowEl.appendChild(button);
    }
    keyboard.appendChild(rowEl);
  }
}

function bind() {
  $("filter").addEventListener("input", (event) => {
    renderList(event.target.value);
  });
  $("filter").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      const word = state.shown[0];
      if (word) selectAnswer(word);
    }
  });
  $("answer-list").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-word]");
    if (!button) return;
    selectAnswer(button.dataset.word);
  });
  $("help-btn").addEventListener("click", () => $("help-dialog").showModal());
  $("help-close").addEventListener("click", () => $("help-dialog").close());
}

function renderList(query) {
  const needle = query.trim().toLowerCase();
  state.shown = state.answers.filter((word) => word.includes(needle));
  const list = $("answer-list");
  const fragment = document.createDocumentFragment();
  for (const word of state.shown) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "answer";
    button.dataset.word = word;
    button.textContent = word;
    button.setAttribute("role", "option");
    button.setAttribute("aria-selected", word === state.selected ? "true" : "false");
    fragment.appendChild(button);
  }
  list.replaceChildren(fragment);
  const noun = state.shown.length === 1 ? "answer" : "answers";
  $("list-count").textContent = `${state.shown.length} ${noun}`;
  const selected = list.querySelector('[aria-selected="true"]');
  if (selected) selected.scrollIntoView({ block: "nearest" });
}

function selectAnswer(word) {
  state.selected = word;
  for (const button of $("answer-list").querySelectorAll(".answer")) {
    button.setAttribute("aria-selected", button.dataset.word === word ? "true" : "false");
  }
  const selected = $("answer-list").querySelector('[aria-selected="true"]');
  if (selected) selected.scrollIntoView({ block: "nearest" });
  play(word);
}

async function play(word) {
  const run = ++state.run;
  const guesses = state.solutions[word];
  if (!guesses) {
    toast("That word is not in the answer list");
    return;
  }
  clearBoard();
  $("status").textContent = `Solving ${word}`;
  $("live").textContent = `Solving ${word}`;
  for (let row = 0; row < guesses.length; row += 1) {
    if (run !== state.run) return;
    await revealRow(row, guesses[row], word);
  }
  if (run !== state.run) return;
  const noun = guesses.length === 1 ? "guess" : "guesses";
  const solved = guesses[guesses.length - 1] === word;
  $("status").textContent = solved
    ? `Solved ${word} in ${guesses.length} ${noun}`
    : `Missed ${word} after ${guesses.length} ${noun}`;
  $("live").textContent = $("status").textContent;
}

function clearBoard() {
  for (const row of $("board").children) {
    for (const tile of row.children) {
      tile.textContent = "";
      tile.dataset.state = "empty";
      tile.className = "tile";
      tile.style.removeProperty("--i");
      tile.setAttribute("aria-label", "empty");
    }
  }
  for (const key of Object.values(state.keys)) delete key.dataset.state;
}

function revealRow(row, guess, target) {
  const marks = scoreGuess(guess, target);
  for (let col = 0; col < COLS; col += 1) {
    const tile = tileAt(row, col);
    tile.textContent = guess[col];
    tile.dataset.state = "tbd";
  }
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    for (let col = 0; col < COLS; col += 1) paint(row, col, guess[col], marks[col]);
    return Promise.resolve();
  }
  for (let col = 0; col < COLS; col += 1) {
    const tile = tileAt(row, col);
    tile.style.setProperty("--i", String(col));
    tile.classList.add("flip");
    window.setTimeout(() => paint(row, col, guess[col], marks[col]), col * STAGGER_MS + FLIP_MS / 2);
  }
  return wait(STAGGER_MS * (COLS - 1) + FLIP_MS + 40);
}

function paint(row, col, letter, mark) {
  const tile = tileAt(row, col);
  tile.dataset.state = mark;
  tile.setAttribute("aria-label", `${letter} ${mark}`);
  const key = state.keys[letter];
  if (!key) return;
  const previous = key.dataset.state;
  if (!previous || RANK[mark] > RANK[previous]) key.dataset.state = mark;
}

function tileAt(row, col) {
  return $("board").children[row].children[col];
}

let toastTimer = 0;
function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => el.classList.remove("show"), 1500);
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
