"""Build the TwiBot-20 training tensors (see Twibot-20/README.md for the raw files).

Scripts from https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN/twibot_20
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
for script in ["preprocess_1.py", "preprocess_2.py"]:
    print(f"Running {script}")
    subprocess.run([sys.executable, script], cwd=here, check=True)
