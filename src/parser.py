import onnx 
import onnxruntime as ort
import networkx as nx 
import numpy as np 
from scipy import linalg

def build_ir(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    
    ir = nx.DiGraph()
    
    # Create a mapping of tensor names to initializers (weights)
    initializer_map = {}
    for initializer in graph.initializer:
        weights = onnx.numpy_helper.to_array(initializer)
        initializer_map[initializer.name] = weights
    
    # Add input nodes
    for inp in graph.input:
        if inp.name not in initializer_map:  # Don't add weights as inputs
            ir.add_node(inp.name, phi={'type': 'Input'})
    
    # Add output nodes
    for output in graph.output:
        ir.add_node(output.name, phi={'type': 'Output'})
    
    # Add operator nodes with phi and psi
    for node in graph.node:
        phi = {
            'type': node.op_type,
            'params': {attr.name: attr.f for attr in node.attribute if attr.HasField('f')}
        }
        
        # Calculate psi (weight properties) if node has weights
        psi = {}
        weight_inputs = [inp for inp in node.input if inp in initializer_map]
        
        if weight_inputs:
            # first weight input, we can consider here. 
            weights = initializer_map[weight_inputs[0]]
            
            # Handle different weight shapes
            if weights.ndim == 2:  # Fully connected: [out, in]
                spectral_norm = np.max(linalg.svdvals(weights))
                sing_vals = linalg.svdvals(weights)
                cond_num = sing_vals[0] / sing_vals[-1] if sing_vals[-1] > 1e-10 else 1e10
                
            elif weights.ndim == 4:  # Conv: [out_ch, in_ch, kh, kw]
                # Reshape to 2D: [out_ch, in_ch * kh * kw]
                reshaped = weights.reshape(weights.shape[0], -1)
                spectral_norm = np.max(linalg.svdvals(reshaped))
                sing_vals = linalg.svdvals(reshaped)
                cond_num = sing_vals[0] / sing_vals[-1] if sing_vals[-1] > 1e-10 else 1e10
                
                # Also store kernel size for Lipschitz computation (Section 2.4)
                phi['params']['kernel_h'] = weights.shape[2]
                phi['params']['kernel_w'] = weights.shape[3]
                
            else:
                spectral_norm = 1.0
                cond_num = 1.0
            
            psi = {
                'spectral_norm': float(spectral_norm),
                'cond_num': float(cond_num),
                'weight_shape': weights.shape
            }
        
        ir.add_node(node.name, phi=phi, psi=psi if psi else {})
    
    # Add edges (dataflow)
    for node in graph.node:
        for inp in node.input:
            if inp in ir.nodes:  # Only add edge if input node exists
                ir.add_edge(inp, node.name)
        for out in node.output:
            ir.add_edge(node.name, out)
    
    return ir
