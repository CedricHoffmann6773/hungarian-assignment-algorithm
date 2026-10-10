import unittest

from hungarian_assignment_algorithm import hungarian, AssignmentError


class TestHungarian(unittest.TestCase):
    def test_square_unique(self):
        m = [[4, 1, 3], [2, 5, 6], [7, 8, 9]]
        self.assertEqual(hungarian(m), [(0, 1), (1, 0), (2, 2)])

    def test_single_element(self):
        self.assertEqual(hungarian([[7]]), [(0, 0)])

    def test_two_by_two(self):
        self.assertEqual(hungarian([[1, 2], [3, 4]]), [(0, 0), (1, 1)])

    def test_more_rows_than_columns(self):
        # 3 rows, 2 columns. Only 2 assignments possible.
        m = [[10, 1], [1, 10], [10, 10]]
        r = hungarian(m)
        self.assertEqual(len(r), 2)
        self.assertEqual(sorted(r), r)
        cols = [c for _, c in r]
        self.assertEqual(sorted(cols), [0, 1])
        # Cheapest two: row0->col1 (1), row1->col0 (1).
        self.assertEqual(set(r), {(0, 1), (1, 0)})

    def test_more_columns_than_rows(self):
        m = [[1, 2, 3], [4, 5, 6]]
        r = hungarian(m)
        self.assertEqual(len(r), 2)
        self.assertEqual(sorted(r), r)
        cols = [c for _, c in r]
        self.assertEqual(sorted(cols), [0, 1])

    def test_rectangular_picks_minimal_total(self):
        m = [[3, 1, 1, 2], [1, 3, 2, 1]]
        r = hungarian(m)
        self.assertEqual(len(r), 2)
        rows = [a for a, _ in r]
        self.assertEqual(sorted(rows), [0, 1])
        cols = [c for _, c in r]
        self.assertEqual(sorted(cols), [0, 1])
        # Best total cost is 2 (row0->col1=1, row1->col0=1 etc.)
        total = sum(m[i][j] for i, j in r)
        self.assertEqual(total, 2)

    def test_zero_cost(self):
        m = [[0, 0], [0, 0]]
        r = hungarian(m)
        self.assertEqual(len(r), 2)
        self.assertEqual(sorted(r), r)
        self.assertEqual([c for _, c in r], [0, 1])

    def test_large_diagonal(self):
        m = [[10, 20], [20, 10]]
        self.assertEqual(hungarian(m), [(0, 0), (1, 1)])

    def test_three_by_three_known(self):
        m = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
        self.assertEqual(hungarian(m), [(0, 0), (1, 1), (2, 2)])

    def test_large_ones_matrix(self):
        m = [[1] * 4 for _ in range(4)]
        r = hungarian(m)
        self.assertEqual(len(r), 4)
        self.assertEqual([a for a, _ in r], [0, 1, 2, 3])
        self.assertEqual([c for _, c in r], [0, 1, 2, 3])

    def test_empty_raises(self):
        with self.assertRaises(AssignmentError):
            hungarian([])
        with self.assertRaises(AssignmentError):
            hungarian([[]])

    def test_negative_raises(self):
        with self.assertRaises(AssignmentError):
            hungarian([[-1, 2], [2, 1]])

    def test_ragged_raises(self):
        with self.assertRaises(AssignmentError):
            hungarian([[1, 2], [3]])

    def test_result_is_list_of_tuples(self):
        r = hungarian([[1, 2], [3, 4]])
        self.assertIsInstance(r, list)
        for pair in r:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)


if __name__ == '__main__':
    unittest.main()
