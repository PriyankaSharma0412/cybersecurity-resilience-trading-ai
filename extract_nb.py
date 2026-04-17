import json

nb = json.load(open(r'c:\Code\Updated_Financial_AI_Robustness_Evaluation.ipynb', 'r', encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    cell_type = c['cell_type']
    source = ''.join(c['source'])
    # Only show first 600 chars of each cell
    print(f'--- CELL {i} ({cell_type}) ---')
    print(source[:600])
    print()
