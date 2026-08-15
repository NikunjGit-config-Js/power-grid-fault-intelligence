import json, csv, os
from pathlib import Path

ROOT = Path(r'C:\Users\ASUS\Downloads\power-grid-fault-intelligence')

print('=' * 60)
print('FINAL VERIFICATION')
print('=' * 60)

# 1. model_comparison.csv unchanged (56 rows)
with open(ROOT / 'results' / 'model_comparison.csv', newline='') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
assert len(rows) == 56, f'Expected 56 rows, got {len(rows)}'
print(f'\n1. model_comparison.csv: {len(rows)} rows -- PASS')

# 2. Figures exist (>=8)
fig_dir = ROOT / 'results' / 'comprehensive_evaluation' / 'figures'
figures = list(fig_dir.glob('*.png'))
print(f'\n2. Figures: {len(figures)} PNGs -- {"PASS" if len(figures) >= 8 else "FAIL"}')
for f in sorted(figures):
    print(f'   {f.name}')

# 3. comprehensive_findings.md exists with required sections
findings = (ROOT / 'results' / 'comprehensive_evaluation' / 'comprehensive_findings.md').read_text(encoding='utf-8')
required_sections = [
    'Executive Summary',
    'Test 1',  # matches "Test 1 → Test 4 Progression"
    'Cross-Test Comparison',
    'Test 4 Detailed Evaluation',
    'Per-Class Performance',
    'Fault Detection Assessment',
    'False Alarm Assessment',
    'CV Stability',
    'Near-Perfect Score Investigation Summary',
    'Computational',
    'Feature Importance',
    'Candidate Shortlist',
    'Limitations',
    'Handoff to Best Model Selection',
]
missing = [s for s in required_sections if s not in findings]
print(f'\n3. comprehensive_findings.md sections: {len(required_sections) - len(missing)}/{len(required_sections)} -- {"PASS" if not missing else "FAIL"}')
if missing:
    for s in missing:
        print(f'   MISSING: {s}')

# 4. No locked test set data loaded
nb = json.load(open(ROOT / 'results' / 'comprehensive_evaluation' / 'comprehensive_evaluation.ipynb'))
all_src = ''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
has_locked_load = 'X_test_hold =' in all_src or 'y_test_hold =' in all_src
print(f'\n4. Locked test set not loaded: {"PASS" if not has_locked_load else "FAIL"}')

# 5. No final model declared (exclude negation lines)
lines_no_neg = [l for l in all_src.split('\n') if 'does not' not in l.lower() and 'no winner' not in l.lower() and 'not selected' not in l.lower() and 'not declare' not in l.lower()]
clean_src = '\n'.join(lines_no_neg)
has_winner = 'selected model' in clean_src.lower() or 'the winner' in clean_src.lower() or 'recommended model' in clean_src.lower()
print(f'\n5. No final model declared: {"PASS" if not has_winner else "FAIL"}')

# 6. Feature importance deferred
has_fi_deferred = 'Feature Importance' in findings and 'Deferred' in findings
print(f'\n6. Feature importance deferred: {"PASS" if has_fi_deferred else "FAIL"}')

# 7. ROC-AUC/PR-AUC documented as N/A
has_n_a = 'N/A' in findings and 'ROC-AUC' in findings
print(f'\n7. ROC-AUC/PR-AUC documented as N/A: {"PASS" if has_n_a else "FAIL"}')

# 8. Notebook syntax
import ast
errs = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        try:
            ast.parse(''.join(c['source']))
        except SyntaxError:
            errs += 1
print(f'\n8. Notebook syntax: {"PASS" if errs == 0 else "FAIL"} ({errs} errors)')

# 9. Candidate shortlist present, no winner
has_shortlist = 'Candidate Shortlist' in findings or 'shortlist' in findings.lower()
has_no_winner = 'does NOT select' in findings or 'NOT final model selection' in findings
print(f'\n9. Candidate shortlist (no winner): {"PASS" if has_shortlist and has_no_winner else "FAIL"}')

print('\n' + '=' * 60)
print('VERIFICATION COMPLETE')
print('=' * 60)
