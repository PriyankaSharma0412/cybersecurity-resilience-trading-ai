"""
Extract key results from the notebook outputs for report writing.
"""
import json
import re

nb = json.load(open('Updated_Financial_AI_Robustness_Evaluation.ipynb', encoding='utf-8'))

# Helper: get text output from a cell
def get_text_output(cell):
    results = []
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'stream':
            results.append(''.join(out.get('text', [])))
        elif out.get('output_type') == 'execute_result':
            data = out.get('data', {})
            if 'text/plain' in data:
                results.append(''.join(data['text/plain']))
    return '\n'.join(results)

def get_html_output(cell):
    for out in cell.get('outputs', []):
        if out.get('output_type') == 'execute_result':
            data = out.get('data', {})
            if 'text/html' in data:
                return ''.join(data['text/html'])
    return ''

# Get all code cells with their source and outputs
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

print("="*70)
print("EXTRACTING KEY RESULTS FROM NOTEBOOK")
print("="*70)

# -------------------------------------------------------
# 1. DATASET INFO
# -------------------------------------------------------
print("\n\n" + "="*70)
print("1. DATASET INFO")
print("="*70)

for i, cell in code_cells:
    src = ''.join(cell['source'])
    out = get_text_output(cell)
    
    # Shape of data
    if 'data.shape' in src and 'yf.download' not in src:
        print(f"\n[Cell {i}] Data shape output:")
        print(out[:500])
    
    if 'yf.download' in src:
        print(f"\n[Cell {i}] Download tickers:")
        print(src[:400])
        print("Output:", out[:300])
    
    if 'feature_cols' in src and 'print' in src and len(out) > 0:
        if 'feature' in out.lower() or 'cols' in out.lower() or len(out) < 1000:
            print(f"\n[Cell {i}] Feature columns:")
            print(out[:800])

# -------------------------------------------------------
# 2. BASELINE MODELS - Anomaly Counts
# -------------------------------------------------------
print("\n\n" + "="*70)
print("2. BASELINE MODELS - ANOMALY COUNTS")
print("="*70)

for i, cell in code_cells:
    src = ''.join(cell['source'])
    out = get_text_output(cell)
    
    # Isolation Forest counts
    if 'Anomaly_IF' in src and ('sum' in src or 'agg' in src) and out:
        print(f"\n[Cell {i}] Isolation Forest anomaly counts:")
        print(out[:600])
    
    # SVM counts
    if 'Anomaly_SVM' in src and ('sum' in src or 'agg' in src or 'count' in src) and out:
        print(f"\n[Cell {i}] SVM anomaly counts:")
        print(out[:600])
    
    # Autoencoder counts  
    if 'Anomaly_AE' in src and ('sum' in src or 'agg' in src or 'count' in src) and out:
        print(f"\n[Cell {i}] Autoencoder anomaly counts:")
        print(out[:600])
    
    # Any unified baseline comparison
    if 'if_split_counts' in out or 'Anomaly_Rate' in out or 'Split' in out:
        if len(out) > 10:
            print(f"\n[Cell {i}] Split anomaly rates:")
            print(out[:800])

# -------------------------------------------------------
# 3. ROBUSTNESS
# -------------------------------------------------------
print("\n\n" + "="*70)
print("3. ROBUSTNESS RESULTS")
print("="*70)

for i, cell in code_cells:
    src = ''.join(cell['source'])
    out = get_text_output(cell)
    
    if ('perturbation' in src.lower() or 'noise' in src.lower() or 'jaccard' in src.lower()) and out and len(out) > 20:
        print(f"\n[Cell {i}] Robustness / perturbation output:")
        print(out[:1000])
    
    if 'stress' in src.lower() and out and len(out) > 20:
        print(f"\n[Cell {i}] Stress scenario output:")
        print(out[:1000])
    
    if 'label_flip' in src or 'label flip' in src.lower() or 'flips' in src.lower():
        if out:
            print(f"\n[Cell {i}] Label flips:")
            print(out[:600])

# -------------------------------------------------------
# 4. EXPLAINABILITY - SHAP
# -------------------------------------------------------
print("\n\n" + "="*70)
print("4. SHAP EXPLAINABILITY")
print("="*70)

for i, cell in code_cells:
    src = ''.join(cell['source'])
    out = get_text_output(cell)
    
    if 'shap' in src.lower() and ('mean' in src.lower() or 'importance' in src.lower() or 'rank' in src.lower()) and out:
        print(f"\n[Cell {i}] SHAP importance:")
        print(out[:1000])

# -------------------------------------------------------
# 5. BEFORE vs AFTER / IMPROVED MODEL
# -------------------------------------------------------
print("\n\n" + "="*70)
print("5. MODEL COMPARISON / IMPROVED MODEL")
print("="*70)

for i, cell in code_cells:
    src = ''.join(cell['source'])
    out = get_text_output(cell)
    
    if ('comparison' in src.lower() or 'unified' in src.lower() or 'baseline' in src.lower()) and out and len(out) > 30:
        print(f"\n[Cell {i}] Comparison output:")
        print(out[:1200])

# -------------------------------------------------------
# 6. EARLY WARNING / EVALUATION METRICS
# -------------------------------------------------------
print("\n\n" + "="*70)
print("6. EARLY WARNING MODEL - METRICS")
print("="*70)

for i, cell in code_cells:
    src = ''.join(cell['source'])
    out = get_text_output(cell)
    
    if ('precision' in out.lower() or 'recall' in out.lower() or 'f1' in out.lower() or 'confusion' in src.lower()) and len(out) > 30:
        print(f"\n[Cell {i}] Metrics:")
        print(out[:1500])

print("\n\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)
