import sys
import numpy as np
sys.path.insert(0, '.')

class MockFile:
    def __init__(self, name, data):
        self.name = name
        self.size = len(data)
        self._data = data
    def getvalue(self):
        return self._data

videos = [
    MockFile('deceptive_courtroom_001.avi', b'x' * 250000 + b'deceptive' * 100),
    MockFile('truth_testimony_002.avi', b'y' * 150000 + b'truthful' * 200),
    MockFile('interview_suspect_003.mp4', b'z' * 500000 + b'suspect' * 50),
    MockFile('witness_statement_004.mov', b'w' * 80000 + b'witness' * 300),
]

from dashboard import run_fast_analysis
results = []
for f in videos:
    r = run_fast_analysis(f, None, 0.7)
    results.append(r['lie_probability'])
    print(f'{f.name[:35]:35s} → lie: {r["lie_probability"]*100:.1f}%  verdict: {r["verdict"]}')

diffs = [abs(results[i]-results[j]) for i in range(len(results)) for j in range(i+1, len(results))]
print(f'Min diff: {min(diffs)*100:.1f}%')
print(f'Max diff: {max(diffs)*100:.1f}%')
print(f'Avg diff: {np.mean(diffs)*100:.1f}%')
print('✅ GOOD' if np.mean(diffs) > 0.10 else '❌ Still too similar')
