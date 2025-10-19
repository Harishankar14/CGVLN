from collections import defaultdict
import numpy as np
import networkx as nx

class AbstractState:
    """
    Stores abstract interpretation results for three domains:
    - Robustness: interval abstractions for adversarial perturbations
    - Precision: affine arithmetic for numerical errors
    - Privacy: taint scores for training data leakage
    """
    def __init__(self):
        self.robustness = defaultdict(lambda: [0.0, 0.0])  # [min, max] intervals
        self.precision = defaultdict(lambda: {'nominal': 0.0, 'errors': {}})  # Affine form
        self.privacy = defaultdict(lambda: 0.0)  # Taint scores


def abstract_interpret(ir: nx.DiGraph):
    """
    Perform abstract interpretation using three specialized domains (Section 2.2) I guess i need some modifcation in the paper.
    Returns AbstractState with vulnerability analysis results
    """
    state = AbstractState()
    
    # Check if graph is DAG
    if not nx.is_directed_acyclic_graph(ir):
        print("WARNING: Graph contains cycles, attempting to continue...")
        # For debugging - find cycles
        try:
            cycles = list(nx.simple_cycles(ir))
            print(f"Found {len(cycles)} cycle(s)")
        except:
            pass
    
    try:
        topo_order = list(nx.topological_sort(ir))
    except:
        # Fallback: use nodes in arbitrary order
        print("WARNING: Could not perform topological sort, using node order")
        topo_order = list(ir.nodes())
    
    # Initialize: Mark inputs as tainted sources for privacy analysis
    for node in topo_order:
        phi = ir.nodes[node].get('phi', {})
        if phi.get('type') == 'Input':
            state.privacy[node] = 1.0  # Tainted source
            state.robustness[node] = [0.0, 1.0]  # Assume normalized input [0,1]
            state.precision[node] = {'nominal': 0.5, 'errors': {}}
    
    # Forward pass: propagate abstract values through operators
    for node in topo_order:
        phi = ir.nodes[node].get('phi', {})
        psi = ir.nodes[node].get('psi', {})
        op_type = phi.get('type', '')
        
        predecessors = list(ir.predecessors(node))
        if not predecessors:
            continue
        
        '''
        With the aspect of Robustness.
        I have added other func as well. 
        extended the model useage included, RESNET50,EFFICIENTNET-V2, MOBILENETV2.
        '''
        if op_type == 'Relu':
            # ReLU: exact bound computation
            inp_min, inp_max = state.robustness[predecessors[0]]
            state.robustness[node] = [max(0.0, inp_min), max(0.0, inp_max)]
        
        elif op_type in ['Sigmoid', 'Tanh']:
            # Sigmoid/Tanh: piecewise linear approximation #D i HAVE DERIVED IT LOCALLY !! 
            inp_min, inp_max = state.robustness[predecessors[0]]
            if op_type == 'Sigmoid':
                out_min = 1.0 / (1.0 + np.exp(-inp_min))
                out_max = 1.0 / (1.0 + np.exp(-inp_max))
            else:  # Tanh
                out_min = np.tanh(inp_min)
                out_max = np.tanh(inp_max)
            state.robustness[node] = [out_min, out_max]
        
        elif op_type == 'Conv' or op_type == 'Gemm' or op_type == 'MatMul':
            # Linear layers: amplify by spectral norm
            inp_min, inp_max = state.robustness[predecessors[0]]
            spectral_norm = psi.get('spectral_norm', 1.0)
            state.robustness[node] = [inp_min * spectral_norm, inp_max * spectral_norm]
        
        elif op_type == 'BatchNormalization':
            # BN multiplies uncertainty by (σ + ε_BN)^-1
            inp_min, inp_max = state.robustness[predecessors[0]]
            epsilon = phi.get('params', {}).get('epsilon', 1e-5)
            # Simplified: assume variance ~1, so amplification factor is ~1/epsilon
            amplification = 1.25  # As per paper Section 2.3
            state.robustness[node] = [inp_min * amplification, inp_max * amplification]
        
        elif op_type == 'Add':
            # Addition: combine intervals
            if len(predecessors) >= 2:
                min1, max1 = state.robustness[predecessors[0]]
                min2, max2 = state.robustness[predecessors[1]]
                state.robustness[node] = [min1 + min2, max1 + max2]
        
        else:
            # Default: propagate from first input
            if predecessors:
                state.robustness[node] = state.robustness[predecessors[0]]
        
        # === Precision Domain (D_P) ===
        if op_type == 'Add':
            # Affine arithmetic: add nominals and merge errors
            if len(predecessors) >= 2:
                nom1 = state.precision[predecessors[0]]['nominal']
                nom2 = state.precision[predecessors[1]]['nominal']
                err1 = state.precision[predecessors[0]]['errors']
                err2 = state.precision[predecessors[1]]['errors']
                
                # Merge error terms
                merged_errors = {**err1}
                for key, val in err2.items():
                    merged_errors[key] = merged_errors.get(key, 0) + val
                
                state.precision[node] = {
                    'nominal': nom1 + nom2,
                    'errors': merged_errors
                }
        
        elif op_type in ['Conv', 'Gemm', 'MatMul']:
            # Multiplication amplifies errors
            if predecessors:
                nom = state.precision[predecessors[0]]['nominal']
                spectral_norm = psi.get('spectral_norm', 1.0)
                state.precision[node] = {
                    'nominal': nom * spectral_norm,
                    'errors': {f'{node}_weight': spectral_norm * 0.01}  # Add weight error
                }
        
        else:
            # Default: propagate precision
            if predecessors:
                state.precision[node] = state.precision[predecessors[0]].copy()
        
        # === Privacy Domain (D_PR) ===
        # Propagate taint through high-sensitivity layers
        cond_num = psi.get('cond_num', 1.0)
        if cond_num > 1000 and predecessors:
            max_taint = max(state.privacy[pred] for pred in predecessors if pred in state.privacy)
            state.privacy[node] = max_taint + 0.5  # Amplify taint
        elif predecessors:
            # Default: propagate maximum taint from inputs
            max_taint = max(state.privacy[pred] for pred in predecessors if pred in state.privacy)
            state.privacy[node] = max_taint
    
    # Apply vulnerability propagation model
    state = propagate_vulnerabilities(ir, state)
    
    return state


def propagate_vulnerabilities(ir: nx.DiGraph, state: AbstractState, max_iter=100, tol=0.001):
    """
    Vulnerability propagation model (Section 2.3)
    Analyzes how architectural patterns amplify risks through layer composition
    """
    try:
        topo_order = list(nx.topological_sort(ir))
    except:
        topo_order = list(ir.nodes())
    
    total_depth = len(topo_order)
    depth_threshold = int(0.7 * total_depth)
    
    # Initialize vulnerability state vector [r, n, p] for each node
    vuln_state = {}
    for node in topo_order:
        r_val = (state.robustness[node][1] - state.robustness[node][0]) / 2.0
        n_val = state.precision[node]['nominal']
        p_val = state.privacy[node]
        vuln_state[node] = [r_val, n_val, p_val]
    
    # Iterative propagation until convergence
    for iteration in range(max_iter):
        prev_state = {node: vuln.copy() for node, vuln in vuln_state.items()}
        
        for idx, node in enumerate(topo_order):
            phi = ir.nodes[node].get('phi', {})
            psi = ir.nodes[node].get('psi', {})
            op_type = phi.get('type', '')
            predecessors = list(ir.predecessors(node))
            
            r, n, p = vuln_state[node]
            
            # ReLU Amplification (Section 2.3)
            if op_type == 'Relu' and predecessors:
                w_norm = psi.get('spectral_norm', 1.0)
                r = prev_state[predecessors[0]][0] * max(1.0, w_norm)
            
            # BN Degradation (Section 2.3)
            if op_type == 'BatchNormalization':
                r = r * 1.25
            
            # Skip Connection Leakage (Section 2.3)
            if op_type == 'Add' and len(predecessors) >= 2:
                # Check if this is in final 30% of network
                if idx > depth_threshold:
                    # Check if connecting distant layers (skip connection)
                    pred_indices = [topo_order.index(pred) for pred in predecessors if pred in topo_order]
                    if pred_indices and (max(pred_indices) - min(pred_indices) > 5):
                        p = max(prev_state[pred][2] for pred in predecessors) + 2.5
            
            # Numerical Cascade (Section 2.3)
            if n > 0.5:  # Threshold θ_n
                for succ in ir.successors(node):
                    if succ in vuln_state:
                        vuln_state[succ][1] = max(vuln_state[succ][1], 0.8 * n)
            
            vuln_state[node] = [r, n, p]
        
        # Check convergence
        max_diff = 0.0
        for node in topo_order:
            for i in range(3):
                diff = abs(vuln_state[node][i] - prev_state[node][i])
                max_diff = max(max_diff, diff)
        
        if max_diff < tol:
            break
    
    # Update state with propagated vulnerabilities
    for node in topo_order:
        r, n, p = vuln_state[node]
        state.robustness[node] = [0.0, r * 2.0]  # Convert back to interval
        state.precision[node]['nominal'] = n
        state.privacy[node] = p
    
    return state
