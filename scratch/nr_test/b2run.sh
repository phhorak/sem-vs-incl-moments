#!/bin/bash
# Run a command in light-2509-fornax, retrying the setup until its python3 is on PATH.
for i in 1 2 3 4 5; do
  source /cvmfs/belle.cern.ch/tools/b2setup >/dev/null 2>&1
  b2setup light-2509-fornax >/dev/null 2>&1
  [[ $(command -v python3) == /cvmfs/*externals* ]] && exec "$@"
  sleep 3
done
echo "b2setup failed" >&2; exit 1
