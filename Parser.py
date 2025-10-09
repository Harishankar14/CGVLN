'''
Second step would be nothing but to Parse computational Graph and 
Build an IR 

What would be the goal of this: Convert ONNX/TF GraphDef to your IR: 𝐼𝑅 = (𝑉, 𝐸, Φ, Ψ)

𝑉: Operator nodes (e.g., ReLU, Conv)
𝐸: Dataflow edges (tensors between ops).
Φ: Semantics (e.g., activation type, norm params)
Ψ: Weight props (e.g., spectral norm = max singular value, condition number = max/min singular values)
'''
import onnx 
import onnxruntime as ort
import networkx as nx 
import numpy as np 
from scipy import linalg

def build_ir(model_path):
    model=onnx.load(model_path)
    graph=model.graph
    
    '''Simple algo for the declaration 
    class DiGraph(
        incoming_graph_data: Any | None (for now Any)
        **attr: Any
    )
    '''
    ir = nx.DiGraph() #Initialize a graph with edges, name, or graph attributes.
    for node in graph.node:
        phi = {'type': node.op_type, 'params': {attr.name: attr.f for attr in node.attribute if attr.HasField('f')}}
        ir.add_node(node.name, phi=phi)
        
    #We can go for adding the edge. 
    '''
    Based on what || how can we push it ?? 
    pseudo algo would be 
    
     1. if input is present in graph 
            then what we can do is add the node, classify the type as Input.
            
    2. If the output is pressent in graph 
            then in the similar way, add the node, classify the type as Output.
            
    3. Followed by the addition of Edge.
        This would be done, once we addd the input and output to the graph. 
        So what i meant to say is, it will iterate the nodes added, then move on to thed edges.
    '''
    
    for input in graph.input:
        ir.add_node(input.name,phi={'type':'Input'})
    
    for output in graph.output:
        ir.add_node(output.name,phi={'type':'Output'})
    for node in graph.node:
        for inp in node.input:
            ir.add_edge(inp,node.name)
        for out in node.output:
            ir.add_edge(node.name,out)
            
    '''
    Now I guess, we can move on to calculating the props for layers
    with weights 
    '''
    '''
    I guess, i have to comeback again for the calculation part, here, for the Weight Props.
    But yes, in a rough way, the math is Mathing, so no props ;) 
    '''
    for initializer in graph.initializer:
        weights = onnx.numpy_helper.to_array(initializer)
        spectral_norm = np.max(linalg.svdvals(weights))
        cond_num = spectral_norm / np.min(linalg.svdvals(weights)) if weights.ndim > 1 else 1.0
        
        for node in ir.nodes:
            if initializer.name in ir.nodes[node].get('phi', {}).get('weights', []):  # Link weights to nodes
                ir.nodes[node]['psi'] = {'spectral_norm': spectral_norm, 'cond_num': cond_num}
    return ir

