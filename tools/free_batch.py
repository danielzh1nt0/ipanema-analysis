import subprocess, sys
for s in sys.argv[1:]: print("=== running", s, flush=True); subprocess.run([sys.executable, s])
