import random
import unittest

from heartbeat import simulate, matmul_ref, allclose
from heartbeat.cli import main


def rand(r, c, rng):
    return [[rng.randint(-5, 5) for _ in range(c)] for _ in range(r)]


class TestCorrectness(unittest.TestCase):
    def test_random_shapes_both_dataflows(self):
        rng = random.Random(1)
        for _ in range(200):
            m, k, n = (rng.randint(1, 7) for _ in range(3))
            A, B = rand(m, k, rng), rand(k, n, rng)
            rows = rng.choice([None, rng.randint(1, 5)])
            cols = rng.choice([None, rng.randint(1, 5)])
            ref = matmul_ref(A, B)
            for df in ("os", "ws"):
                with self.subTest(df=df, m=m, k=k, n=n, rows=rows, cols=cols):
                    res = simulate(A, B, df, rows, cols)
                    self.assertTrue(allclose(res.C, ref))
                    self.assertEqual(res.macs, m * k * n)
                    self.assertLessEqual(res.utilization, 1.0)

    def test_floats(self):
        rng = random.Random(2)
        A = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(5)]
        B = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(4)]
        for df in ("os", "ws"):
            self.assertTrue(allclose(simulate(A, B, df).C, matmul_ref(A, B)))


class TestTiming(unittest.TestCase):
    def test_output_stationary_cycles(self):
        A, B = rand(4, 6, random.Random(3)), rand(6, 5, random.Random(4))
        res = simulate(A, B, "os")
        self.assertEqual(res.cycles, 6 + 4 + 5 - 2)
        self.assertEqual(res.load_cycles, 0)

    def test_weight_stationary_cycles(self):
        A, B = rand(4, 6, random.Random(3)), rand(6, 5, random.Random(4))
        res = simulate(A, B, "ws")
        self.assertEqual(res.cycles, 4 + 6 + 5 - 2)
        self.assertEqual(res.load_cycles, 6)

    def test_tiling_adds_up(self):
        A, B = rand(4, 3, random.Random(5)), rand(3, 4, random.Random(6))
        res = simulate(A, B, "os", rows=2, cols=2)
        self.assertEqual(res.tiles, 4)
        self.assertEqual(res.cycles, 4 * (3 + 2 + 2 - 2))

    def test_trace_has_one_frame_per_cycle(self):
        A, B = rand(3, 3, random.Random(7)), rand(3, 3, random.Random(8))
        for df in ("os", "ws"):
            res = simulate(A, B, df, trace=True)
            self.assertEqual(len(res.trace), res.cycles)

    def test_wavefront_first_and_last_pe(self):
        # In OS, PE(0,0) works on cycle 0; the far corner first works on cycle r+c-2.
        A, B = rand(3, 4, random.Random(9)), rand(4, 3, random.Random(10))
        tr = simulate(A, B, "os", trace=True).trace
        self.assertTrue(tr[0].active[0][0])
        self.assertFalse(any(s.active[2][2] for s in tr[:4]))
        self.assertTrue(tr[4].active[2][2])


class TestErrorsAndCli(unittest.TestCase):
    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            simulate([[1, 2]], [[1, 2]])
        with self.assertRaises(ValueError):
            simulate([[1]], [[1]], "xyz")

    def test_cli_runs(self):
        self.assertEqual(main(["run", "--a", "1,2;3,4", "--b", "5,6;7,8", "--trace"]), 0)
        self.assertEqual(main(["compare", "--m", "5", "--k", "4", "--n", "6", "--rows", "2", "--cols", "3"]), 0)
        self.assertEqual(main(["run", "--a", "1,2", "--b", "1,2"]), 2)


if __name__ == "__main__":
    unittest.main()
