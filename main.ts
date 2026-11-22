// sokoban-solver-01 entry: a push-optimal Sokoban solver.
//
// Every level is a hallway with one to four rooms hanging off it, each through a single door. A box pushed into the
// hallway could never come back (the player would have to stand in the wall on the hallway's far side), so each box
// stays in its room and each room is a Sokoban level of its own: a level's optimal push count is the sum of its rooms'
// optimal push counts. This program splits the level into rooms, solves every room push-optimally, and stitches the
// room plans together with the player's walks along the hallway.
//
// Inside a room only pushes count, so the search runs over push states: the boxes plus the area the player can walk to
// with those boxes in place, canonicalised by its lowest square. A* with an admissible, consistent estimate — the
// cheapest assignment of boxes to goals over single-box push distances — is optimal, and dead squares and frozen
// blocks prune states no push can save. Work, never the clock, bounds the search, so two runs answer alike. A room
// whose share of the work runs out is searched again on its own account; if even that found nothing, its boxes are
// pushed one at a time, in every assignment and order, which solves some rooms the full search could not reach.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The whole run's work, in search states expanded: roughly 15 to 35 thousand a second, so far inside the wall.
const TOTAL_WORK = 300_000;
// One room's search never creates more states than this, whatever its share of the work: a memory guard.
const STATE_LIMIT = 400_000;
// A room is always allowed this much work, even when the pool is nearly gone.
const MIN_SHARE = 1_000;
// A room whose share ran out gets one search of its own, this large: the generator keeps only rooms its own push
// search settles within 250,000 states, and every room met here settled in under two thousand expansions.
const FINAL_WORK = 250_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTERS = ['u', 'd', 'l', 'r'];
const BIG = 1e9;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Room {
  area: Uint8Array; // squares the player may walk on: the room's floor, its door and the hallway
  floor: Uint8Array; // squares a box may stand on: the room's floor and its door, never the hallway
  goals: number[];
  boxes: number[];
  start: number; // the hallway square just outside the room's door
}

interface Plan {
  pushes: number;
  moves: string;
  player: number;
  boxes: number[];
}

interface SearchResult {
  plan: Plan | null;
  pops: number; // states expanded: the work the search spent
}

/** A level in XSB: walls, goals, boxes and the player's square. */
function parseLevel(text: string): Level {
  const rows = text.replace(/\r/g, '').split('\n').filter((line) => line.length > 0);
  const h = rows.length;
  const w = Math.max(1, ...rows.map((line) => line.length));
  const size = w * h;
  const wall = new Uint8Array(size).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    const line = rows[y]!;
    for (let x = 0; x < line.length; x++) {
      const ch = line[x]!;
      if (ch === '#') continue;
      const i = y * w + x;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  // A board is enclosed: the frame is wall whatever a short row left there, so no walk leaves the board.
  for (let x = 0; x < w; x++) {
    wall[x] = 1;
    wall[(h - 1) * w + x] = 1;
  }
  for (let y = 0; y < h; y++) {
    wall[y * w] = 1;
    wall[y * w + w - 1] = 1;
  }
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The level's rooms: the parts of the floor above and below the player's row (the hallway) that hold boxes, each with
 * its door and the hallway square outside it. Null when the level is not shaped that way.
 */
function splitRooms(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  if (player < 0) return null;
  const hall = Math.floor(player / w);
  const comp = new Int32Array(size).fill(-1);
  const parts: { cells: number[]; boxes: number[]; goals: number[] }[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || Math.floor(s / w) === hall || comp[s] >= 0) continue;
    const id = parts.length;
    const cells = [s];
    comp[s] = id;
    for (let i = 0; i < cells.length; i++) {
      const p = cells[i]!;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (!wall[q] && Math.floor(q / w) !== hall && comp[q] < 0) {
          comp[q] = id;
          cells.push(q);
        }
      }
    }
    parts.push({ cells, boxes: [], goals: [] });
  }
  for (const b of boxes) {
    if (Math.floor(b / w) === hall || comp[b] < 0) return null;
    parts[comp[b]!]!.boxes.push(b);
  }
  for (const g of goals) {
    if (Math.floor(g / w) === hall || comp[g] < 0) return null;
    parts[comp[g]!]!.goals.push(g);
  }
  const rooms: Room[] = [];
  for (const part of parts) {
    if (part.boxes.length === 0) {
      if (part.goals.length > 0) return null;
      continue;
    }
    if (part.boxes.length !== part.goals.length) return null;
    const area = new Uint8Array(size);
    const floor = new Uint8Array(size);
    for (const c of part.cells) {
      area[c] = 1;
      floor[c] = 1;
    }
    for (let x = 0; x < w; x++) if (!wall[hall * w + x]) area[hall * w + x] = 1;
    let start = -1;
    for (const c of part.cells) {
      const y = Math.floor(c / w);
      if (y !== hall - 1 && y !== hall + 1) continue;
      const outside = y === hall - 1 ? c + w : c - w;
      if (area[outside]) {
        start = outside;
        break;
      }
    }
    if (start < 0) return null;
    rooms.push({
      area,
      floor,
      goals: part.goals.sort((a, b) => a - b),
      boxes: part.boxes.sort((a, b) => a - b),
      start,
    });
  }
  return rooms;
}

/** The squares from which a box alone can still be pushed onto one of the room's goals, pulled back from the goals. */
function liveSquares(w: number, room: Room): Uint8Array {
  const size = room.floor.length;
  const live = new Uint8Array(size);
  const queue: number[] = [];
  for (const g of room.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const y = x - d;
      if (room.floor[y] && room.area[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Every order of 0 to n - 1, flattened: the assignments and orders a matching or a plan may use. */
function permutations(n: number): Int32Array {
  const out: number[] = [];
  const used = new Uint8Array(n);
  const order: number[] = [];
  const walk = (): void => {
    if (order.length === n) {
      for (const g of order) out.push(g);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = 1;
      order.push(i);
      walk();
      order.pop();
      used[i] = 0;
    }
  };
  walk();
  return Int32Array.from(out);
}

/** A box's sorted squares with `at` swapped for `to`, written into `out`. */
function moved(boxes: number[], at: number, to: number, out: number[]): number[] {
  let j = 0;
  for (let i = 0; i < boxes.length; i++) if (i !== at) out[j++] = boxes[i]!;
  out[j] = to;
  while (j > 0 && out[j - 1]! > out[j]!) {
    const t = out[j - 1]!;
    out[j - 1] = out[j]!;
    out[j] = t;
    j--;
  }
  return out;
}

/** The key of a state: its region's lowest square and its boxes, in base 256 (a board holds fewer than 256 squares). */
function keyOf(boxes: number[], low: number): number {
  let key = low;
  for (let i = 0; i < boxes.length; i++) key = key * 256 + boxes[i]!;
  return key;
}

/** The shortest walk from `from` to `to` around walls and `blocked` boxes, in lower LURD letters, or null if none. */
function walkTo(wall: Uint8Array, blocked: Uint8Array, w: number, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how: string[] = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (wall[q] || blocked[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = LETTERS[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  const out: string[] = [];
  for (let p = to; p !== from; p = prev[p]!) out.push(how[p]!);
  return out.reverse().join('');
}

/** The plan's moves replayed from `player` and `boxes`: the pushes it makes and the player's square after it. */
function replayPlan(level: Level, plan: Plan, player: number, boxes: number[]): Plan | null {
  const wall = level.wall;
  const w = level.w;
  const occupied = new Uint8Array(wall.length);
  for (const b of boxes) occupied[b] = 1;
  let p = player;
  let pushes = 0;
  for (let i = 0; i < plan.moves.length; i++) {
    const ch = plan.moves[i]!;
    const k = ch === 'u' ? 0 : ch === 'd' ? 1 : ch === 'l' ? 2 : 3;
    const d = DY[k]! * w + DX[k]!;
    const next = p + d;
    if (wall[next]) return null;
    if (occupied[next]) {
      const beyond = next + d;
      if (wall[beyond] || occupied[beyond]) return null;
      occupied[next] = 0;
      occupied[beyond] = 1;
      pushes++;
    }
    p = next;
  }
  const out: number[] = [];
  for (let i = 0; i < occupied.length; i++) if (occupied[i]) out.push(i);
  return { pushes, moves: plan.moves, player: p, boxes: out };
}

/**
 * One room solved push-optimally from the hallway square outside its door, within `budget` expanded states, or the
 * best it found before that: null when the search gives up or the room is unsolvable. The estimate is the cheapest
 * assignment of boxes to goals over the room's single-box push distances: no plan can beat it, and one push lowers it
 * by at most one, so A* settles every state at its optimal push count.
 */
function solveRoom(level: Level, room: Room, budget: number, stateLimit: number): SearchResult {
  const w = level.w;
  const size = level.wall.length;
  const startBoxes = room.boxes;
  const n = startBoxes.length;
  if (n === 0) return { plan: { pushes: 0, moves: '', player: room.start, boxes: [] }, pops: 0 };
  if (n > 6) return { plan: null, pops: 0 };

  const goal = new Uint8Array(size);
  for (const g of room.goals) goal[g] = 1;
  const live = liveSquares(w, room);
  for (const b of startBoxes) if (!live[b]) return { plan: null, pops: 0 };

  // The push distance from every square to each goal, a box alone on the board: the estimate's table.
  const goalDist: Int32Array[] = [];
  const queue: number[] = [];
  for (const g of room.goals) {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    queue.length = 0;
    queue.push(g);
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const y = x - d;
        if (room.floor[y] && room.area[y - d] && dist[y] < 0) {
          dist[y] = dist[x]! + 1;
          queue.push(y);
        }
      }
    }
    goalDist.push(dist);
  }
  const perms = permutations(n);
  const permCount = perms.length / n;
  const estimate = (boxes: number[]): number => {
    let least = BIG;
    for (let p = 0; p < permCount; p++) {
      const base = p * n;
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = goalDist[perms[base + i]!]![boxes[i]!]!;
        if (d < 0) {
          sum = BIG;
          break;
        }
        sum += d;
      }
      if (sum < least) least = sum;
    }
    return least;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // The squares the player reaches from `from` with the boxes in `occupied`, marked in `marks` and returning the lowest.
  const reach = (from: number, marks: Int32Array, at: number): number => {
    let low = from;
    let top = 0;
    stack[top++] = from;
    marks[from] = at;
    while (top > 0) {
      const p = stack[--top]!;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (room.area[q] && !occupied[q] && marks[q] !== at) {
          marks[q] = at;
          stack[top++] = q;
        }
      }
    }
    return low;
  };

  const boxStates: number[][] = [];
  const playerStates: number[] = [];
  const parents: number[] = [];
  const depths: number[] = [];
  const heurs: number[] = [];
  const keys: number[] = [];
  const best = new Map<number, number>();
  const heap: number[] = [];

  const before = (a: number, b: number): boolean => {
    const fa = depths[a]! + heurs[a]!;
    const fb = depths[b]! + heurs[b]!;
    if (fa !== fb) return fa < fb;
    if (heurs[a] !== heurs[b]) return heurs[a]! < heurs[b]!;
    return a < b;
  };
  const push = (idx: number): void => {
    let i = heap.length;
    heap.push(idx);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const pop = (): number => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && before(heap[c + 1]!, heap[c]!)) c++;
        if (!before(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  for (const b of startBoxes) occupied[b] = 1;
  const firstKey = keyOf(startBoxes, reach(room.start, seen, ++stamp));
  for (const b of startBoxes) occupied[b] = 0;
  const firstH = estimate(startBoxes);
  if (firstH >= BIG) return { plan: null, pops: 0 };
  boxStates.push(startBoxes);
  playerStates.push(room.start);
  parents.push(-1);
  depths.push(0);
  heurs.push(firstH);
  keys.push(firstKey);
  best.set(firstKey, 0);
  push(0);

  let done = -1;
  let pops = 0;
  while (heap.length > 0 && pops < budget && done < 0) {
    const idx = pop();
    if (best.get(keys[idx]!) !== idx) continue;
    pops++;
    // A zero estimate means every box stands on a goal (the assignment is perfect and all distances are zero).
    if (heurs[idx] === 0) {
      done = idx;
      break;
    }
    const boxes = boxStates[idx]!;
    const depth = depths[idx]! + 1;
    for (let i = 0; i < n; i++) occupied[boxes[i]!] = 1;
    reach(playerStates[idx]!, seen, ++stamp);
    const here = stamp;
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        if (!room.floor[t] || occupied[t] || !live[t] || seen[b - d] !== here) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        let dead = false;
        for (let c = 0; c < 4; c++) {
          const a = t + (c === 0 ? 0 : c === 1 ? -1 : c === 2 ? -w : -w - 1);
          const b2 = a + 1;
          const c2 = a + w;
          const d2 = c2 + 1;
          if (!(level.wall[a] || occupied[a]) || !(level.wall[b2] || occupied[b2])) continue;
          if (!(level.wall[c2] || occupied[c2]) || !(level.wall[d2] || occupied[d2])) continue;
          if ((occupied[a] && !goal[a]) || (occupied[b2] && !goal[b2]) || (occupied[c2] && !goal[c2]) || (occupied[d2] && !goal[d2])) {
            dead = true;
            break;
          }
        }
        if (!dead) {
          const next = moved(boxes, bi, t, new Array<number>(n));
          const h = estimate(next);
          if (h < BIG) {
            const key = keyOf(next, reach(b, mark, ++stamp));
            const old = best.get(key);
            if ((old === undefined || depths[old]! > depth) && boxStates.length < stateLimit) {
              const id = boxStates.length;
              boxStates.push(next);
              playerStates.push(b);
              parents.push(idx);
              depths.push(depth);
              heurs.push(h);
              keys.push(key);
              best.set(key, id);
              push(id);
            }
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
      }
    }
    for (let i = 0; i < n; i++) occupied[boxes[i]!] = 0;
  }
  if (done < 0) return { plan: null, pops };

  // Walk the state chain back to front, emitting each push's walk and push letter.
  const chain: number[] = [];
  for (let s = done; s >= 0; s = parents[s]!) chain.push(s);
  chain.reverse();
  const blocked = new Uint8Array(size);
  let boxes = startBoxes;
  let player = room.start;
  let moves = '';
  for (let i = 1; i < chain.length; i++) {
    const was = boxStates[chain[i - 1]!]!;
    const now = boxStates[chain[i]!]!;
    let from = -1;
    let to = -1;
    for (const b of was) if (!now.includes(b)) from = b;
    for (const b of now) if (!was.includes(b)) to = b;
    const d = to - from;
    for (const b of was) blocked[b] = 1;
    const walk = walkTo(level.wall, blocked, w, player, from - d);
    for (const b of was) blocked[b] = 0;
    if (walk === null) return { plan: null, pops };
    const k = d === -w ? 0 : d === w ? 1 : d === -1 ? 2 : 3;
    moves += walk + LETTERS[k];
    player = from;
    boxes = now;
  }
  return { plan: { pushes: chain.length - 1, moves, player, boxes: boxes.slice() }, pops };
}

/**
 * Pushes one box to `target` around the other boxes, which stand still, over the fewest pushes: the walk to each
 * push's square and the push letter, or null when no push path exists.
 */
function pushOneBox(level: Level, room: Room, boxes: number[], bi: number, target: number, player: number, live: Uint8Array): Plan | null {
  const w = level.w;
  const size = level.wall.length;
  const occupied = new Uint8Array(size);
  for (let i = 0; i < boxes.length; i++) occupied[boxes[i]!] = 1;

  const seen = new Int32Array(size);
  const marks = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const reachFrom = (from: number, at: number, into: Int32Array): number => {
    let low = from;
    let top = 0;
    stack[top++] = from;
    into[from] = at;
    while (top > 0) {
      const p = stack[--top]!;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (room.area[q] && !occupied[q] && into[q] !== at) {
          into[q] = at;
          stack[top++] = q;
        }
      }
    }
    return low;
  };

  const startBox = boxes[bi]!;
  if (startBox === target) return { pushes: 0, moves: '', player, boxes: boxes.slice() };
  const pos: number[] = [startBox];
  const atPlayer: number[] = [player];
  const parent: number[] = [-1];
  const stepDir: number[] = [0];
  const visited = new Map<number, number>();
  visited.set(startBox * 256 + reachFrom(player, ++stamp, seen), 0);
  let found = -1;
  for (let head = 0; head < pos.length && found < 0; head++) {
    const b = pos[head]!;
    occupied.fill(0);
    for (let i = 0; i < boxes.length; i++) if (i !== bi) occupied[boxes[i]!] = 1;
    occupied[b] = 1;
    reachFrom(atPlayer[head]!, ++stamp, seen);
    const here = stamp;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const t = b + d;
      if (!room.floor[t] || occupied[t] || !live[t] || seen[b - d] !== here) continue;
      occupied[b] = 0;
      occupied[t] = 1;
      const key = t * 256 + reachFrom(b, ++stamp, marks);
      occupied[t] = 0;
      occupied[b] = 1;
      if (visited.has(key)) continue;
      visited.set(key, pos.length);
      pos.push(t);
      atPlayer.push(b);
      parent.push(head);
      stepDir.push(k);
      if (t === target) {
        found = pos.length - 1;
        break;
      }
    }
  }
  if (found < 0) return null;

  const chain: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) chain.push(s);
  chain.reverse();
  const blocked = new Uint8Array(size);
  let p = player;
  let moves = '';
  for (const s of chain) {
    const from = pos[parent[s]!]!;
    const to = pos[s]!;
    const d = to - from;
    for (let i = 0; i < boxes.length; i++) if (i !== bi) blocked[boxes[i]!] = 1;
    blocked[from] = 1;
    const walk = walkTo(level.wall, blocked, w, p, from - d);
    for (let i = 0; i < boxes.length; i++) if (i !== bi) blocked[boxes[i]!] = 0;
    blocked[from] = 0;
    if (walk === null) return null;
    moves += walk + LETTERS[stepDir[s]!];
    p = from;
  }
  return { pushes: chain.length, moves, player: p, boxes: boxes.slice() };
}

/**
 * A room solved without optimality: every assignment of goals to boxes and every order of the boxes is tried, each
 * box pushed to its goal around the boxes already placed. Null when none of them works.
 */
function solveRoomGreedy(level: Level, room: Room): Plan | null {
  const n = room.boxes.length;
  if (n === 0) return { pushes: 0, moves: '', player: room.start, boxes: [] };
  if (n > 6) return null;
  const live = liveSquares(level.w, room);
  for (const b of room.boxes) if (!live[b]) return null;
  const assign = permutations(n);
  const order = permutations(n);
  const count = assign.length / n;
  let attempts = 0;
  for (let a = 0; a < count; a++) {
    for (let o = 0; o < count; o++) {
      if (++attempts > 2_000) return null;
      const boxes = room.boxes.slice();
      let player = room.start;
      let moves = '';
      let pushes = 0;
      let ok = true;
      for (let step = 0; step < n && ok; step++) {
        const bi = order[o * n + step]!;
        const target = room.goals[assign[a * n + bi]!]!;
        if (boxes[bi] === target) continue;
        const part = pushOneBox(level, room, boxes, bi, target, player, live);
        if (part === null) {
          ok = false;
          break;
        }
        moves += part.moves;
        player = part.player;
        pushes += part.pushes;
        boxes[bi] = target;
      }
      if (!ok) continue;
      if (boxes.every((b) => room.goals.includes(b))) {
        return { pushes, moves, player, boxes: boxes.slice() };
      }
    }
  }
  return null;
}

/** One level: every room solved (optimally when its share of the work allows), then stitched along the hallway. */
function solveLevel(level: Level, workCap: number): { moves: string; used: number } {
  const rooms = splitRooms(level);
  if (rooms === null || rooms.length === 0) return { moves: '', used: 0 };
  const plans: (Plan | null)[] = new Array<Plan | null>(rooms.length).fill(null);
  let used = 0;
  let left = workCap;
  let remaining = rooms.length;
  const failed: number[] = [];
  for (let i = 0; i < rooms.length; i++) {
    const share = Math.max(MIN_SHARE, Math.floor(left / Math.max(1, remaining)));
    const result = solveRoom(level, rooms[i]!, share, STATE_LIMIT);
    used += result.pops;
    left = Math.max(0, left - result.pops);
    remaining--;
    if (result.plan !== null) plans[i] = result.plan;
    else failed.push(i);
  }
  // A room whose share ran out gets one search of its own, past the pool: no room the generator keeps can spend
  // anything like this much, so an answered instance stays answered and stays optimal.
  const still: number[] = [];
  for (const i of failed) {
    const result = solveRoom(level, rooms[i]!, FINAL_WORK, STATE_LIMIT);
    used += result.pops;
    if (result.plan !== null) plans[i] = result.plan;
    else still.push(i);
  }
  // Only a room past even that guarantee is left: a one-box-at-a-time plan, however many pushes it takes.
  for (const i of still) {
    const plan = solveRoomGreedy(level, rooms[i]!);
    if (plan === null) return { moves: '', used };
    plans[i] = plan;
  }

  // Stitch: walk to each room's door, run its plan, and walk out again on the way to the next.
  let boxes = level.boxes.slice();
  let player = level.player;
  let moves = '';
  for (let i = 0; i < rooms.length; i++) {
    const room = rooms[i]!;
    const plan = plans[i]!;
    if (player !== room.start) {
      const blocked = new Uint8Array(level.wall.length);
      for (const b of boxes) blocked[b] = 1;
      const walk = walkTo(level.wall, blocked, level.w, player, room.start);
      if (walk === null) return { moves: '', used };
      moves += walk;
      player = room.start;
    }
    const checked = replayPlan(level, plan, player, boxes);
    if (checked === null) return { moves: '', used };
    moves += plan.moves;
    boxes = checked.boxes;
    player = checked.player;
  }
  return { moves, used };
}

const ids = readdirSync(INPUT_DIR).sort();
let pool = TOTAL_WORK;
for (let i = 0; i < ids.length; i++) {
  const id = ids[i]!;
  const level = parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  const cap = Math.max(MIN_SHARE, Math.floor(pool / Math.max(1, ids.length - i)));
  const result = solveLevel(level, cap);
  pool = Math.max(0, pool - result.used);
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${result.moves}\n`);
}
