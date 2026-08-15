findings = open(r'C:\Users\ASUS\Downloads\power-grid-fault-intelligence\results\comprehensive_evaluation\comprehensive_findings.md', 'r', encoding='utf-8').read()
lines = [l.strip() for l in findings.split('\n') if l.startswith('##')]
for l in lines:
    print(l)
