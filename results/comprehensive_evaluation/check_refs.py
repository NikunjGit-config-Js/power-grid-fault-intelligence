import json
nb = json.load(open(r'C:\Users\ASUS\Downloads\power-grid-fault-intelligence\results\comprehensive_evaluation\comprehensive_evaluation.ipynb'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if 'locked' in src.lower() or 'split_indices' in src.lower():
        print(f'Cell {i} ({c["cell_type"]}):')
        for line in src.split('\n'):
            if 'locked' in line.lower() or 'split_indices' in line.lower():
                print(f'  {line.strip()[:120]}')
    if 'selected model' in src.lower() or 'winner' in src.lower() or 'recommended model' in src.lower():
        print(f'Cell {i} ({c["cell_type"]}): CONTAINS winner/selected/recommended')
        for line in src.split('\n'):
            if 'selected model' in line.lower() or 'winner' in line.lower() or 'recommended model' in line.lower():
                print(f'  {line.strip()[:120]}')
