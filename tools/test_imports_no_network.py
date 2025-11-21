import importlib
import os
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

modules = [
    'with_api.runner_api',
    'with_api.api_client',
    'with_api.gemini_client',
    'agent.api_client',
    'agent.gemini_client',
    'agent.summarizer_hybrid',
]

print('Running import smoke tests (no network)')
for m in modules:
    try:
        importlib.import_module(m)
        print('OK:', m)
    except Exception as e:
        print('ERR:', m, '->', type(e).__name__, e)
