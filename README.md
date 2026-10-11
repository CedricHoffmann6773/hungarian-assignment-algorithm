# hungarian-assignment-algorithm

Minimum-cost one-to-one assignment between two sets via the Kuhn-Munkres augmenting-path method.

```python
from hungarian_assignment_algorithm import hungarian

pairs = hungarian([
    [4, 1, 3],
    [2, 5, 6],
    [7, 8, 9],
])
# [(0, 1), (1, 0), (2, 2)]
```

`hungarian(cost)` takes a non-empty matrix (list of lists) of non-negative numeric costs and returns a list of `(row, col)` tuples forming a minimum-cost perfect matching on the smaller dimension.

## Why this exists

You have N workers and M jobs, a cost for each pair, and need the cheapest one-to-one pairing. `scipy.optimize.linear_sum_assignment` solves this, but pulling SciPy in for one algorithm is heavy and not always available. This is a pure-Python implementation with no dependencies.

The trade-off: correctness and zero dependencies over raw speed. This is O(N^3) but pure Python, so it is fine up to a few hundred workers per side; beyond that, call SciPy.

## What the reader will hit

- **Rectangular matrices.** For a 3x2 matrix you get 2 pairs; the unpaired row is not reported. If you need to know which rows or columns went unassigned, take the row/column index sets and subtract.
- **Negative costs raise `AssignmentError`.** The algorithm pads rectangular matrices with zero-cost dummy slots; negative costs would make padding distort the optimum, so they are rejected up front. If your costs can be negative, add a constant offset to make them all non-negative before calling.
- **Ties.** When multiple assignments have equal total cost, the result is deterministic but follows the algorithm's natural row-then-column scan order. Do not read any meaning into *which* tie was chosen.
