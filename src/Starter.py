import onnx 
import networkx as nx
from ir_builder import build_ir
from domains import abstract_interpret
from vulnerability_decetor import detect_vulnerabilities, generate_report

def main():
    print("=" * 80)
    print("CGVulnScan - Static Vulnerability Analysis for Neural Networks")
    print("=" * 80)
    
    # Load model
    model_path = "/home/harishankar/Music/cgvulnscan/EfficientNet-V2-s.onnx"
    print(f"\n[1/4] Loading model: {model_path}")
    
    try:
        model = onnx.load(model_path)
        graph = model.graph 
        print(f"✓ Model loaded: {len(graph.node)} operators, {len(graph.input)} inputs")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Build IR
    print(f"\n[2/4] Building intermediate representation...")
    try:
        ir = build_ir(model_path)
        print(f"✓ IR constructed: {ir.number_of_nodes()} nodes, {ir.number_of_edges()} edges")
        print(f"  - Is DAG: {nx.is_directed_acyclic_graph(ir)}")
    except Exception as e:
        print(f"✗ Failed to build IR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run abstract interpretation
    print(f"\n[3/4] Running abstract interpretation with vulnerability propagation...")
    try:
        state = abstract_interpret(ir)
        print(f"✓ Abstract interpretation complete")
    except Exception as e:
        print(f"✗ Failed during abstract interpretation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Sample vulnerability states
    print(f"\n--- Sample Vulnerability States (first 5 nodes) ---")
    node_list = list(ir.nodes())[:5]
    for node in node_list:
        phi = ir.nodes[node].get('phi', {})
        print(f"\n{node} ({phi.get('type', 'Unknown')})")
        print(f"  Robustness: {state.robustness[node]}")
        print(f"  Precision:  {state.precision[node]['nominal']:.4f}")
        print(f"  Privacy:    {state.privacy[node]:.4f}")
    
    # Detect vulnerabilities
    print(f"\n[4/4] Detecting operator-level vulnerabilities...")
    try:
        vulnerabilities = detect_vulnerabilities(ir, state)
        print(f"✓ Vulnerability detection complete")
    except Exception as e:
        print(f"✗ Failed during vulnerability detection: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Print summary
    print("\n" + "=" * 80)
    print("VULNERABILITY SUMMARY")
    print("=" * 80)
    
    total = sum(len(v) for v in vulnerabilities.values())
    print(f"\nTotal Issues Found: {total}")
    print(f"  • Adversarial:  {len(vulnerabilities['adversarial'])}")
    print(f"  • Numerical:    {len(vulnerabilities['numerical'])}")
    print(f"  • Privacy:      {len(vulnerabilities['privacy'])}")
    
    # Show critical/high severity issues
    print("\n--- Critical & High Severity Issues ---")
    
    for category in ['adversarial', 'numerical', 'privacy']:
        critical_high = [v for v in vulnerabilities[category] 
                        if v.get('severity') in ['critical', 'high']]
        
        if critical_high:
            print(f"\n{category.upper()}:")
            for vuln in critical_high[:3]:  # Show first 3
                print(f"  [{vuln['severity'].upper()}] {vuln['type']}")
                print(f"    Node: {vuln['node']}")
                print(f"    {vuln['message']}")
    
    # Show some examples by type
    print("\n--- Example Vulnerabilities ---")
    
    if vulnerabilities['adversarial']:
        print("\n[ADVERSARIAL EXAMPLE]")
        vuln = vulnerabilities['adversarial'][0]
        print(f"Type: {vuln['type']}")
        print(f"Node: {vuln['node']}")
        print(f"Severity: {vuln['severity']}")
        print(f"Message: {vuln['message']}")
        if 'lipschitz' in vuln:
            print(f"Lipschitz: {vuln['lipschitz']:.2e}")
    
    if vulnerabilities['numerical']:
        print("\n[NUMERICAL EXAMPLE]")
        vuln = vulnerabilities['numerical'][0]
        print(f"Type: {vuln['type']}")
        print(f"Node: {vuln['node']}")
        print(f"Severity: {vuln['severity']}")
        print(f"Message: {vuln['message']}")
    
    if vulnerabilities['privacy']:
        print("\n[PRIVACY EXAMPLE]")
        vuln = vulnerabilities['privacy'][0]
        print(f"Type: {vuln['type']}")
        print(f"Node: {vuln['node']}")
        print(f"Severity: {vuln['severity']}")
        print(f"Message: {vuln['message']}")
    
    # Generate detailed report
    print("\n" + "=" * 80)
    print("Generating detailed report...")
    try:
        generate_report(vulnerabilities, 'resnet50_vulnerability_report.txt')
        print("✓ Full report saved to: resnet50_vulnerability_report.txt")
    except Exception as e:
        print(f"✗ Failed to generate report: {e}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
