# ----------------------------------------------------------------
# nsga2.py  —  non-dominated sorting and crowding (Deb et al. 2002).
# Objective vectors are (score to MAXIMISE, cost to MINIMISE).
# ----------------------------------------------------------------


def dominates(a, b):
    """Pareto dominance for objectives (maximise first, minimise second)."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def nondominated_sort(objs):
    """NSGA-II fast non-dominated sort: list of fronts (lists of indices), best first."""
    n = len(objs)
    dom, count, fronts = [[] for _ in range(n)], [0] * n, [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                dom[p].append(q)
            elif dominates(objs[q], objs[p]):
                count[p] += 1
        if count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dom[p]:
                count[q] -= 1
                if count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(objs, front):
    """NSGA-II crowding distance of every index in one front (boundary points get infinity)."""
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    d = {i: 0.0 for i in front}
    for m in range(2):
        srt = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[srt[0]][m], objs[srt[-1]][m]
        d[srt[0]] = d[srt[-1]] = float("inf")
        if hi == lo:
            continue
        for j in range(1, len(srt) - 1):
            d[srt[j]] += (objs[srt[j + 1]][m] - objs[srt[j - 1]][m]) / (hi - lo)
    return d


def rank_and_crowd(objs):
    """Front rank and crowding distance of every individual (for tournament selection)."""
    rank, crowd = {}, {}
    for r, f in enumerate(nondominated_sort(objs)):
        for i in f:
            rank[i] = r
        crowd.update(crowding(objs, f))
    return rank, crowd


def select_survivors(objs, size):
    """NSGA-II survivor selection: fill `size` slots front by front, breaking the last
    front by crowding distance.
    """
    chosen = []
    for f in nondominated_sort(objs):
        if len(chosen) + len(f) <= size:
            chosen += f
        else:
            cd = crowding(objs, f)
            chosen += sorted(f, key=lambda i: -cd[i])[:size - len(chosen)]
            break
    return chosen
