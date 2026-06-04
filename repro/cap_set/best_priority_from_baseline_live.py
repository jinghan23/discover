import math
import numpy as np

_CACHE = {}

def priority(el, n):
    x = tuple(int(v) % 3 for v in el)
    if n != 8:
        return sum((i + 1) * (v + 1) for i, v in enumerate(x))

    key = ("cap8_rank_v6_swap", n)
    if key not in _CACHE:
        _CACHE[key] = _build_order(n)

    idx = 0
    p = 1
    for v in x:
        idx += v * p
        p *= 3
    return _CACHE[key].get(idx, 0.0)


def _build_order(n):
    N = 3 ** n
    powers = (3 ** np.arange(n, dtype=np.int64))
    ids = np.arange(N, dtype=np.int64)

    D = np.empty((N, n), dtype=np.int8)
    for i in range(n):
        D[:, i] = (ids // int(powers[i])) % 3
    X = D.astype(np.int16)

    comp = np.empty((N, N), dtype=np.uint16)
    for i in range(N):
        comp[i] = (((-X - X[i]) % 3) @ powers).astype(np.uint16)

    def ecap_cols(cols, mode=0):
        A = X[:, cols]
        if mode == 0:
            q = (A[:, 0] * A[:, 0] + A[:, 1] * A[:, 1] + A[:, 2] * A[:, 2] + 2 * A[:, 3] * A[:, 3]) % 3
        elif mode == 1:
            q = (A[:, 0] * A[:, 0] + A[:, 1] * A[:, 2] + A[:, 3] * A[:, 3]) % 3
        elif mode == 2:
            q = (A[:, 0] * A[:, 1] + A[:, 2] * A[:, 3]) % 3
        elif mode == 3:
            q = (A[:, 0] * A[:, 0] + A[:, 1] * A[:, 1] + A[:, 2] * A[:, 3]) % 3
        elif mode == 4:
            q = (A[:, 0] * A[:, 2] + A[:, 1] * A[:, 1] + 2 * A[:, 3] * A[:, 3]) % 3
        elif mode == 5:
            q = (A[:, 0] * A[:, 3] + A[:, 1] * A[:, 2] + A[:, 2] * A[:, 2]) % 3
        else:
            q = (A[:, 0] * A[:, 0] + 2 * A[:, 1] * A[:, 3] + A[:, 2] * A[:, 2]) % 3
        return (q == 0) & np.any(A != 0, axis=1)

    parts = [
        ((0, 1, 2, 3), (4, 5, 6, 7), 0, 150000.0, 1700.0),
        ((0, 1, 4, 5), (2, 3, 6, 7), 0, 24500.0, 1450.0),
        ((0, 2, 4, 6), (1, 3, 5, 7), 0, 24500.0, 1450.0),
        ((0, 3, 4, 7), (1, 2, 5, 6), 0, 24500.0, 1450.0),
        ((0, 1, 6, 7), (2, 3, 4, 5), 0, 23000.0, 1300.0),
        ((0, 2, 5, 7), (1, 3, 4, 6), 0, 23000.0, 1300.0),
        ((0, 3, 5, 6), (1, 2, 4, 7), 0, 23000.0, 1300.0),
        ((0, 4, 2, 6), (1, 5, 3, 7), 0, 21500.0, 1250.0),
        ((0, 5, 2, 7), (1, 4, 3, 6), 0, 21500.0, 1250.0),
        ((0, 6, 1, 7), (2, 4, 3, 5), 0, 21500.0, 1250.0),
        ((0, 7, 1, 6), (2, 5, 3, 4), 0, 21500.0, 1250.0),
        ((0, 1, 2, 4), (3, 5, 6, 7), 1, 8400.0, 640.0),
        ((0, 1, 3, 5), (2, 4, 6, 7), 1, 8400.0, 640.0),
        ((0, 2, 3, 6), (1, 4, 5, 7), 1, 8400.0, 640.0),
        ((0, 4, 5, 6), (1, 2, 3, 7), 1, 8400.0, 640.0),
        ((0, 1, 5, 6), (2, 3, 4, 7), 2, 7800.0, 575.0),
        ((0, 2, 5, 6), (1, 3, 4, 7), 2, 7800.0, 575.0),
        ((0, 3, 4, 6), (1, 2, 5, 7), 2, 7800.0, 575.0),
        ((0, 1, 4, 7), (2, 3, 5, 6), 3, 7300.0, 540.0),
        ((0, 2, 4, 7), (1, 3, 5, 6), 3, 7300.0, 540.0),
        ((0, 3, 5, 7), (1, 2, 4, 6), 4, 7000.0, 520.0),
        ((0, 1, 3, 6), (2, 4, 5, 7), 4, 7000.0, 520.0),
        ((0, 2, 3, 7), (1, 4, 5, 6), 5, 5200.0, 420.0),
        ((0, 4, 6, 7), (1, 2, 3, 5), 6, 5000.0, 400.0),
    ]

    base = np.zeros(N, dtype=np.float64)
    ecaps = {}
    fallback = None

    for j, (a, b, m, wp, ws) in enumerate(parts):
        ea = ecaps.setdefault((a, m), ecap_cols(a, m))
        eb = ecaps.setdefault((b, m), ecap_cols(b, m))
        prod = ea & eb
        if j == 0:
            fallback = prod.copy()
        base += wp * prod
        base += ws * (ea.astype(np.int16) + eb.astype(np.int16))

    wt = np.count_nonzero(X, axis=1)
    ones = np.sum(X == 1, axis=1)
    twos = np.sum(X == 2, axis=1)
    s1 = np.sum(X, axis=1) % 3
    sq = np.sum((X * X) % 3, axis=1) % 3

    base += 72.0 * (4 - np.abs(wt - 4))
    base += 47.0 * (ones == twos)
    base += 18.0 * (np.abs(ones - twos) == 1)
    base += 23.0 * (s1 == 0)
    base += 19.0 * (sq == 1)

    q1 = X[:, 2] % 3 == (X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1]) % 3
    q2 = X[:, 5] % 3 == (X[:, 3] * X[:, 3] + X[:, 4] * X[:, 4]) % 3
    q3 = X[:, 7] % 3 == (X[:, 1] * X[:, 1] + X[:, 6] * X[:, 6]) % 3
    q4 = (X[:, 3] + X[:, 7]) % 3 == ((X[:, 0] + X[:, 4]) ** 2 + (X[:, 2] + X[:, 6]) ** 2) % 3
    q5 = (X[:, 2] + X[:, 4]) % 3 == ((X[:, 0] + X[:, 7]) ** 2 + (X[:, 1] + X[:, 5]) ** 2) % 3
    q6 = (X[:, 1] + X[:, 7]) % 3 == ((X[:, 3] + X[:, 6]) ** 2 + (X[:, 0] + X[:, 5]) ** 2) % 3
    q7 = (X[:, 0] + X[:, 2] + X[:, 5]) % 3 == ((X[:, 3] + X[:, 7]) ** 2 + X[:, 6] * X[:, 6]) % 3
    q8 = (X[:, 4] + X[:, 6] + X[:, 7]) % 3 == ((X[:, 0] + X[:, 1]) ** 2 + X[:, 3] * X[:, 3]) % 3
    q9 = (X[:, 0] + X[:, 4] + X[:, 6]) % 3 == ((X[:, 1] + X[:, 5]) ** 2 + (X[:, 2] + X[:, 7]) ** 2) % 3
    q10 = (X[:, 2] + X[:, 3] + X[:, 7]) % 3 == ((X[:, 0] + X[:, 6]) ** 2 + (X[:, 1] + X[:, 4]) ** 2) % 3
    q11 = (X[:, 0] + X[:, 5] + X[:, 7]) % 3 == ((X[:, 1] + X[:, 3]) ** 2 + X[:, 4] * X[:, 4]) % 3
    q12 = (X[:, 1] + X[:, 4] + X[:, 6]) % 3 == ((X[:, 2] + X[:, 5]) ** 2 + X[:, 0] * X[:, 0]) % 3

    qs = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]
    base += 3400.0 * (q1 & q2 & q3)
    base += 2400.0 * (q1 & q4 & q5)
    base += 1950.0 * (q2 & q4 & q6)
    base += 1450.0 * (q3 & q7 & q8)
    base += 1100.0 * (q5 & q9 & q10)
    base += 900.0 * (q4 & q11 & q12)

    acc = np.zeros(N, dtype=np.int16)
    for q in qs:
        acc += q.astype(np.int16)
    base += 170.0 * acc

    lin = [
        (1, 1, 1, 1, 1, 1, 1, 1, 0, 21.0),
        (1, 2, 0, 1, 2, 0, 1, 2, 1, 16.0),
        (0, 1, 2, 2, 1, 0, 2, 1, 2, 14.0),
        (2, 0, 1, 2, 1, 2, 1, 0, 2, 10.0),
        (1, 2, 2, 1, 0, 1, 2, 0, 0, 8.0),
        (2, 1, 1, 0, 2, 2, 0, 1, 1, 7.0),
        (1, 0, 2, 1, 1, 2, 2, 0, 2, 6.0),
        (2, 2, 1, 0, 1, 0, 2, 1, 0, 5.0),
        (0, 2, 1, 1, 2, 2, 1, 0, 1, 4.0),
        (1, 1, 2, 0, 2, 1, 0, 2, 2, 3.5),
        (2, 1, 0, 1, 0, 2, 2, 1, 1, 3.0),
    ]
    for row in lin:
        c = np.array(row[:8], dtype=np.int16)
        base += row[9] * ((X @ c) % 3 == row[8])

    def make_noise(seed):
        h = ids.astype(np.uint64) ^ np.uint64(seed)
        h ^= h >> np.uint64(15)
        h *= np.uint64(2246822519)
        h ^= h >> np.uint64(13)
        h *= np.uint64(3266489917)
        h ^= h >> np.uint64(16)
        return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)

    def run_greedy(score, k0, k1, k2, k3, m0, m1, m2, m3, second_cut, second_w):
        blocked = np.zeros(N, dtype=bool)
        selected = []
        while True:
            avail = np.flatnonzero(~blocked)
            if len(avail) == 0:
                break

            s = len(selected)
            if s < 100:
                k, mult = k0, m0
            elif s < 220:
                k, mult = k1, m1
            elif s < 360:
                k, mult = k2, m2
            else:
                k, mult = k3, m3
            k = min(k, len(avail))

            cand = avail[np.argpartition(score[avail], -k)[-k:]]
            best_c = int(cand[0])
            best_val = 1.0e100

            for c0 in cand:
                c = int(c0)
                if selected:
                    z = comp[c, selected]
                    free_z = z[~blocked[z]]
                    add = int(len(free_z))
                    second = 0
                    if s > 55 and add:
                        second = int(np.sum(score[free_z] > score[c] - second_cut))
                    val = add * mult + second * second_w - score[c] * 0.001
                else:
                    val = -score[c] * 0.001
                if val < best_val:
                    best_val = val
                    best_c = c

            if selected:
                blocked[comp[best_c, selected]] = True
            blocked[best_c] = True
            selected.append(best_c)
        return selected

    def greedy_extend(start, score):
        selected = [int(v) for v in start]
        blocked = np.zeros(N, dtype=bool)
        for i, v in enumerate(selected):
            if i:
                blocked[comp[v, selected[:i]]] = True
            blocked[v] = True

        while True:
            avail = np.flatnonzero(~blocked)
            if len(avail) == 0:
                break
            if selected:
                vals = score[avail] - 245.0 * np.array(
                    [np.sum(~blocked[comp[int(a), selected]]) for a in avail],
                    dtype=np.float64,
                )
            else:
                vals = score[avail]
            v = int(avail[int(np.argmax(vals))])
            if selected:
                blocked[comp[v, selected]] = True
            blocked[v] = True
            selected.append(v)
        return selected

    def improve_one_for_two(start, score, max_rounds=10):
        selected = [int(v) for v in start]
        for _ in range(max_rounds):
            in_sel = np.zeros(N, dtype=bool)
            in_sel[selected] = True
            sel_arr = np.array(selected, dtype=np.int64)

            buckets = {}
            outside = np.flatnonzero(~in_sel)
            for v0 in outside:
                v = int(v0)
                cc = comp[v, sel_arr]
                hit = sel_arr[in_sel[cc]]
                if len(hit) == 2:
                    a = int(hit[0])
                    b = int(hit[1])
                    buckets.setdefault(a, []).append(v)
                    buckets.setdefault(b, []).append(v)

            made = False
            for u, cand in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
                if len(cand) < 2 or not in_sel[u]:
                    continue
                cand = sorted(set(cand), key=lambda z: -score[z])[:90]
                for i in range(len(cand)):
                    a = cand[i]
                    for j in range(i + 1, len(cand)):
                        b = cand[j]
                        c = int(comp[a, b])
                        if c != u and in_sel[c]:
                            continue
                        selected.remove(u)
                        selected.append(a)
                        selected.append(b)
                        selected = greedy_extend(selected, score)
                        made = True
                        break
                    if made:
                        break
                if made:
                    break
            if not made:
                break
        return selected

    best = np.flatnonzero(fallback).astype(np.uint16).tolist()

    seeds = [
        2166136261, 2246822519, 3266489917, 668265263, 374761393, 1274126177,
        2654435761, 1597334677, 3812015801, 1442695041, 3202034521, 97531,
        40503, 8675309, 1013904223, 747796405, 2891336453, 277803737,
        1103515245, 362437, 521288629, 88675123, 5783321, 704602925,
        31337, 2718281828, 3141592653, 1618033988, 42424243, 69069,
        123456789, 987654321, 4294967291, 1181783497, 12345, 99991,
        19260817, 735632791, 254326997, 60493, 1402946737, 915488749,
        1664525, 22695477, 214013, 134775813, 8121, 2147483647,
        2654435769, 10101, 20202, 30303, 40404, 50505, 60606, 70707,
        80808, 90909, 11235813, 27182818, 31415926, 16180339,
    ]

    params = [
        (52, 92, 150, 280, 1650.0, 1260.0, 930.0, 650.0, 4200.0, 1.45),
        (64, 104, 170, 320, 1580.0, 1210.0, 890.0, 620.0, 3850.0, 1.35),
        (44, 84, 135, 260, 1720.0, 1320.0, 960.0, 690.0, 4550.0, 1.55),
        (72, 120, 190, 360, 1500.0, 1160.0, 850.0, 590.0, 3500.0, 1.25),
        (36, 76, 128, 250, 1840.0, 1400.0, 1010.0, 730.0, 5000.0, 1.70),
        (86, 138, 220, 420, 1420.0, 1100.0, 805.0, 560.0, 3150.0, 1.10),
        (58, 116, 205, 500, 1540.0, 1120.0, 760.0, 510.0, 2950.0, 0.95),
        (96, 150, 250, 560, 1360.0, 1040.0, 735.0, 485.0, 2700.0, 0.85),
    ]

    for i, seed in enumerate(seeds):
        noise = make_noise(seed)
        wobble = make_noise(seed ^ 0x9E3779B9)
        score = base + (18.0 + 7.0 * (i % 9)) * noise + (3.0 + (i % 5)) * (wobble - 0.5) * acc
        sel = run_greedy(score, *params[i % len(params)])
        sel = greedy_extend(sel, score)
        if len(sel) > len(best):
            best = sel

    tail_score = base + 31.0 * make_noise(4242424242)
    ext = greedy_extend(best, tail_score)
    if len(ext) > len(best):
        best = ext

    improved = improve_one_for_two(best, tail_score + 11.0 * make_noise(987654321), 8)
    if len(improved) > len(best):
        best = improved

    rank = {}
    big = 10.0 * N
    for i, v in enumerate(best):
        rank[int(v)] = big - i

    used = set(rank)
    tail = np.argsort(base + 0.001 * make_noise(1357913579))
    r = float(N)
    for v in tail[::-1]:
        iv = int(v)
        if iv not in used:
            rank[iv] = r + base[iv] * 1e-6
            r -= 1.0

    return rank
