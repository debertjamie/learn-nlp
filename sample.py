import json
import random
import sys
 
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
docs = [json.loads(l) for l in open("data/docs.jsonl", encoding="utf-8")]
random.seed(7)
for d in random.sample(docs, n):
    print(f"{d['title']}  --  {d['text'][:110]}")