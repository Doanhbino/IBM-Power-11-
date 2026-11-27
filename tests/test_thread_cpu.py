"""thread_cpu must attribute CPU time per thread and group threads by affinity mask."""
import json, os, subprocess, sys, tempfile, textwrap, time, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from thread_cpu import parse_cpulist, snapshot, measure

CHILD = textwrap.dedent('''
    import os, sys, threading, time
    cpus = [int(c) for c in sys.argv[1:3]]
    def spin(cpu):
        os.sched_setaffinity(0, {cpu})
        end = time.time() + 20
        while time.time() < end: pass
    def idle(cpu):
        os.sched_setaffinity(0, {cpu})
        time.sleep(20)
    threading.Thread(target=spin, args=(cpus[0],), daemon=True).start()
    threading.Thread(target=idle, args=(cpus[1],), daemon=True).start()
    print('ready', flush=True)
    time.sleep(20)
''')


class ThreadCpu(unittest.TestCase):
    def test_parse_cpulist(self):
        self.assertEqual(parse_cpulist('0,32,64-66'), frozenset({0, 32, 64, 65, 66}))
        self.assertEqual(parse_cpulist('7'), frozenset({7}))

    @unittest.skipUnless(len(os.sched_getaffinity(0)) >= 2, 'needs two allowed CPUs')
    def test_busy_and_idle_threads_are_separated_by_mask(self):
        a, b = sorted(os.sched_getaffinity(0))[:2]
        child = subprocess.Popen([sys.executable, '-c', CHILD, str(a), str(b)], stdout=subprocess.PIPE, text=True)
        try:
            child.stdout.readline()
            time.sleep(.3)
            result = measure(child.pid, window=1.0, group_cpus=frozenset({a}))
        finally:
            child.kill(); child.wait()
        self.assertGreater(result['group_cpu_percent'], 60)       # busy thread pinned to `a`
        self.assertLess(result['other_cpu_percent'], 20)          # idle thread and main thread
        self.assertEqual(result['group_threads'], 1)

    def test_snapshot_reads_own_threads(self):
        data = snapshot(os.getpid())
        self.assertTrue(data)
        for tid, row in data.items():
            self.assertIn('allowed', row); self.assertGreaterEqual(row['ticks'], 0)


if __name__ == '__main__':
    unittest.main()
