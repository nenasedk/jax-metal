#!/bin/zsh
# Usage: rea_q.sh TOOL JSON_ARGS  -> prints REA's normalized_result (or raw text).
out=$(python3 -I "${0:A:h}/rea_session.py" call "$1" "$2")
python3 -I -c '
import json,sys
t=sys.stdin.read()
try:
    d=json.loads(t); r=d.get("result",d)
    v=r.get("normalized_result", r.get("raw_result", r))
    print(v if isinstance(v,str) else json.dumps(v,indent=1))
except Exception:
    print(t)' <<< "$out"
