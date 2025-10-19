import onnx 
import networkx as nx 
import numpy as np 
from scipy import linalg

def build_ir(model_path):
    """
    Build IR from ONNX model: IR = (V, E, Φ, Ψ)
    V: Operator nodes
    E: Dataflow edges
    Φ: Operator semantics
    Ψ: Weight properties
    """
    model = onnx.load(model_path)
    graph = model.graph
    
    ir = nx.DiGraph()
    
    #Creating an Mapper here, this is the most important !!
    initializer_map = {}
    for initializer in graph.initializer:
        weights = onnx.numpy_helper.to_array(initializer)   #taking the help of onnx.helper.
        initializer_map[initializer.name] = weights
    
    #next major thing is adding input nodes. (without the weights)
    '''
    A simple algo would be nothing but
      if they are not in intializer_map we can basically add the note.. 
    '''
    for inp in graph.input:
        if inp.name not in initializer_map:
            ir.add_node(inp.name, phi={'type': 'Input'}, psi={})
    
    # operator nodes with Φ (semantics) and Ψ (weight properties)
    for node in graph.node:
        # Build Φ: operator semantics
        phi = {'type': node.op_type,'params': {}}
        
        # Extract attributes (e.g., epsilon for BatchNorm)
        for attr in node.attribute:
            if attr.HasField('f'):
                phi['params'][attr.name] = attr.f
            elif attr.HasField('i'):
                phi['params'][attr.name] = attr.i
        
        # Build Ψ: weight properties (spectral norm, condition number)
        psi = {}
        weight_inputs = [inp for inp in node.input if inp in initializer_map]
        
        if weight_inputs:
            weights = initializer_map[weight_inputs[0]]
            
            try:
                if weights.ndim == 2:  # Fully connected layer
                    sing_vals = linalg.svdvals(weights)
                    spectral_norm = sing_vals[0]
                    cond_num = sing_vals[0] / sing_vals[-1] if sing_vals[-1] > 1e-10 else 1e10
                    
                elif weights.ndim == 4:  # Convolutional layer
                    # Reshape [out_ch, in_ch, kh, kw] -> [out_ch, in_ch*kh*kw]
                    reshaped = weights.reshape(weights.shape[0], -1)
                    sing_vals = linalg.svdvals(reshaped)
                    spectral_norm = sing_vals[0]
                    cond_num = sing_vals[0] / sing_vals[-1] if sing_vals[-1] > 1e-10 else 1e10
                    
                    # Store kernel size for Lipschitz computation
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
            except Exception as e:
                # Handle numerical issues in SVD
                psi = {'spectral_norm': 1.0, 'cond_num': 1.0}
        
        ir.add_node(node.name, phi=phi, psi=psi)
    
    '''
    Now i guess, we can add the output node.. 
    '''
    for output in graph.output:
        if output.name not in ir.nodes:
            ir.add_node(output.name, phi={'type': 'Output'}, psi={})
    
    
    # Map tensors to their producer nodes
    tensor_producers = {}
    for node in graph.node:
        for out_tensor in node.output:
            tensor_producers[out_tensor] = node.name
    
    # Connect consumers to producers
    for node in graph.node:
        for inp_tensor in node.input:
            if inp_tensor in tensor_producers:
                '''
                nothing but mapping the producer node to the consumer. 
                '''
                ir.add_edge(tensor_producers[inp_tensor], node.name)
            elif inp_tensor in ir.nodes:
                # It's a graph input
                ir.add_edge(inp_tensor, node.name)
            # Skip if it's a weight (initializer) - >> yeah i ignored for them. don't care for now .
    
    # Connect final operators to output nodes
    for output in graph.output:
        if output.name in tensor_producers:
            ir.add_edge(tensor_producers[output.name], output.name)
    
    return ir
