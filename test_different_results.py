import sys
sys.path.insert(0,'.')

# Simulate with two different fake file objects
class FakeFile:
    def __init__(self, name, content):
        self.name = name
        self.size = len(content)
        self.type = 'video/mp4'
        self._content = content
    def getvalue(self):
        return self._content

import numpy as np
import hashlib, os, tempfile, json
from datetime import datetime
from dashboard import run_fast_analysis, load_detector

model, _ = load_detector()

# Test with two completely different byte contents
f1 = FakeFile('video1.mp4', b'A'*50000 + b'\x00'*50000)
f2 = FakeFile('video2.mp4', b'\xFF'*50000 + b'\x80'*50000)

r1 = run_fast_analysis(f1, model)
r2 = run_fast_analysis(f2, model)

print(f'Video 1 lie_prob: {r1["lie_probability"]:.3f}')
print(f'Video 2 lie_prob: {r2["lie_probability"]:.3f}')
diff = abs(r1["lie_probability"] - r2["lie_probability"])
print(f'Difference: {diff:.3f}')
print('✅ DIFFERENT' if diff > 0.05 else '❌ TOO SIMILAR')