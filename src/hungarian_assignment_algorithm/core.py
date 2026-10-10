from __future__ import annotations


class AssignmentError(ValueError):
    """Raised when the cost matrix is empty or shape-inconsistent."""


def hungarian(cost):
    """
    Minimum-cost one-to-one assignment via the Kuhn-Munkres algorithm.

    Returns a list of (row, col) tuples with exactly min(R, C) pairs.

    Design decisions, because the brief is ambiguous on several points:
      1. Rectangular matrices are supported by padding with zero columns or
         rows internally. Costs must be non-negative so padding does not distort
         the optimum. Negative costs raise AssignmentError rather than risk a
         silent wrong answer from padding.
      2. Unassigned indices are not returned: for an R x C matrix with R > C,
         some rows simply have no column, and vice versa.
      3. Ties are broken deterministically by lowest row then lowest column,
         because the algorithm scans i, j in natural order. This means the
         result is reproducible across runs.
      4. Cost values are compared with ==, so integer or exact-rational costs
         are the intended use. Floats work but their equality semantics are the
         caller's responsibility.
    """
    if not cost or not cost[0]:
        raise AssignmentError("cost matrix must be non-empty")
    R = len(cost)
    C = len(cost[0])
    for row in cost:
        if len(row) != C:
            raise AssignmentError("ragged matrix")
        for v in row:
            if v < 0:
                raise AssignmentError("negative cost")

    # Work on a square copy padded with zero-cost dummy slots.
    N = R if R > C else C
    M = [[0] * N for _ in range(N)]
    for i in range(R):
        row = cost[i]
        Mi = M[i]
        for j in range(C):
            Mi[j] = row[j]

    # u[i], v[j]: vertex potentials covering rows and columns.
    # p[j]: row assigned to column j, or -1.
    # way[j]: predecessor column for augmenting path reconstruction.
    u = [0] * (N + 1)
    v = [0] * (N + 1)
    p = [-1] * (N + 1)
    way = [-1] * (N + 1)

    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (N + 1)
        used = [False] * (N + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = -1

            for j in range(1, N + 1):
                if used[j]:
                    continue
                # Potential-reduced slack for (i0-1, j-1).
                cur = M[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            # Update potentials so the chosen next column has zero slack.
            for j in range(N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == -1:
                break

        # Walk predecessors to flip the augmenting path.
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    result = []
    for j in range(1, N + 1):
        r = p[j] - 1
        if r < R and (j - 1) < C:
            result.append((r, j - 1))

    result.sort()
    return result
