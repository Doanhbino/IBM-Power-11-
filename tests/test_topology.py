"""Affinity plans must respect irregular cpusets and disjoint worker placement."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inventory import cpu_list
from sweep import build_plan, selected_cpus


class TopologyTests(unittest.TestCase):
    def setUp(self):
        # Two NUMA nodes, each with two cores and two allowed SMT siblings.
        self.rows = [dict(cpu=c, node=n, siblings_allowed=s)
                     for n, pairs in [(0, [(2, 10), (6, 14)]), (3, [(21, 29), (25, 33)])]
                     for pair in pairs for c in pair for s in [list(pair)]]

    def test_irregular_cpuset_no_invented_cpu(self):
        plan = build_plan({'smt_limits': [1, 2], 'layouts': ['all', 'spread', 'numa-local'],
                           'threads': [1, 2, 4, 8], 'workers': [1, 2, 4], 'repeats': 1}, self.rows)
        allowed = {r['cpu'] for r in self.rows}
        for case in plan:
            flat = [c for mask in case['cpu_masks'] for c in mask]
            self.assertEqual(len(flat), len(set(flat)))
            self.assertTrue(set(flat) <= allowed)
            self.assertTrue(all(len(mask) == case['threads'] for mask in case['cpu_masks']))
            if case['membind'] is not None:
                nodes = {r['node'] for r in self.rows if r['cpu'] in flat}
                self.assertEqual(nodes, {case['membind']})

    def test_one_sibling_per_core(self):
        self.assertEqual({r['cpu'] for r in selected_cpus(self.rows, 1)}, {2, 6, 21, 25})

    def test_cpu_range_parse(self):
        self.assertEqual(cpu_list('2,6,10-12,29'), [2, 6, 10, 11, 12, 29])


if __name__ == '__main__':
    unittest.main()
