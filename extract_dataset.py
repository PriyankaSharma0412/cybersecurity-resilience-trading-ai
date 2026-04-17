"""Extract dataset stats and feature columns from notebook."""
import json

nb = json.load(open('Updated_Financial_AI_Robustness_Evaluation.ipynb', encoding='utf-8'))
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

def get_text(cell):
    results = []
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'stream':
            results.append(''.join(out.get('text', [])))
        elif out.get('output_type') == 'execute_result':
            data = out.get('data', {})
            if 'text/plain' in data:
                results.append(''.join(data['text/plain']))
    return '\n'.join(results)

# Print cells 0-35
for i, cell in code_cells:
    if i > 40:
        break
    src = ''.join(cell['source'])
    out = get_text(cell)
    if out.strip():
        print(f"\n--- Cell {i} ---")
        print("SRC:", src[:300].replace('\n',' '))
        print("OUT:", out[:600])
