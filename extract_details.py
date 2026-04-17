"""
Extract dataset info, robustness, and SHAP results from notebook.
"""
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

# Print all outputs per cell for specific cell ranges
target_ranges = {
    "DATASET + FEATURES (cells 5-27)": range(0, 30),
    "ROBUSTNESS (cells 73-90)": range(70, 95),
    "SHAP (cells 91-112)": range(88, 115),
}

for label, rng in target_ranges.items():
    print(f"\n{'='*70}")
    print(label)
    print('='*70)
    for i, cell in code_cells:
        if i not in rng:
            continue
        src = ''.join(cell['source'])
        out = get_text(cell)
        if not out.strip():
            continue
        # Only show relevant cells
        keywords = ['shape', 'feature', 'tickers', 'anomal', 'perturb', 'jaccard',
                    'stress', 'shap', 'precision', 'recall', 'f1', 'flip', 'robust',
                    'rows', 'columns', 'assets', 'split', 'print', 'saved']
        if any(k in src.lower() or k in out.lower() for k in keywords):
            print(f"\n--- Cell {i} ---")
            print("SOURCE (first 200 chars):", src[:200].replace('\n', ' '))
            print("OUTPUT:")
            print(out[:1000])
