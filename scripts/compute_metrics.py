import json
p='pipeline_outputs.json'
with open(p) as f:
    j=json.load(f)
inj=set(j['injected'])
det=set(j['detected'])
a=len(inj)
b=len(det)
TP=len(inj & det)
precision = TP / b if b>0 else 0
recall = TP / a if a>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
print(f"Injected: {a}")
print(f"Detected: {b}")
print(f"True positives: {TP}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
print('\nScores summary:')
for k,v in j.get('scores_summary',{}).items():
    print(f" - {k}: {v}")
