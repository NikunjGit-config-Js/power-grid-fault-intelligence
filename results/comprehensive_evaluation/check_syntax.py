import json, ast
nb = json.load(open(r'C:\Users\ASUS\Downloads\power-grid-fault-intelligence\results\comprehensive_evaluation\comprehensive_evaluation.ipynb'))
errs = 0
code_cells = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        code_cells += 1
        src = ''.join(c['source'])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f'Cell {i}: SYNTAX ERROR: {e}')
            errs += 1
print(f'Total cells: {len(nb["cells"])}')
print(f'Code cells: {code_cells}')
print(f'Syntax errors: {errs}')
