from collections import defaultdict
import numpy as np
import networkx as nx 
'''
So Here for Robustness 
We can use piecewise -> Majorly for Sigmoid. 

Precision:-> v = nominal + ∑ ε_i * e_i (yeah will update it properly in the readme )

I guess then we can propogate it backward from outputs.

'''
class AbstractState:
    def __init__(self):
        self.robustness = defaultdict(lambda: [0, 0])  # [min, max] per tensor
        self.precision = defaultdict(lambda: {'nominal': 0, 'errors': {}})  # Affine form
        self.privacy = defaultdict(lambda: 0)  # Taint score, one of the factor which i can consider i guess, if i consider, then i have to recalculate the math again, in the Parser.py 
'''
Then i guess we can pass an abstract state
further putting it in the form of list and making it iteratable.
'''
def abstract_interpret(ir: nx.DiGraph):
    state = AbstractState()
    topo_order = list(nx.topological_sort(ir))
    for node in topo_order:
        if ir.nodes[node]['phi']['type'] == 'Input':
            state.privacy[node] = 1.0  # Tainted source
    
    for node in topo_order:
        phi = ir.nodes[node]['phi']
        inputs = list(ir.predecessors(node))
        
        '''
        For now added the robustness for Relu only.
        Lets see, how it goes for Relu, then i guess, we can choose a diff actfn
        '''
        if phi['type'] == 'Relu':
            inp_min, inp_max = state.robustness[inputs[0]]
            state.robustness[node] = [max(0, inp_min), max(0, inp_max)]
        
        # Precision:(nohing  much here i guess self explaintory)
        if phi['type'] == 'Add':
            nom1 = state.precision[inputs[0]]['nominal']
            nom2 = state.precision[inputs[1]]['nominal']
            state.precision[node] = {'nominal': nom1 + nom2, 'errors': {**state.precision[inputs[0]]['errors'], **state.precision[inputs[1]]['errors']}}
        
        #I guess we can Propagate taint if sensitivity high  (we must yeah, thing works fine)
        psi = ir.nodes[node].get('psi', {})
        if 'cond_num' in psi and psi['cond_num'] > 1000:
            state.privacy[node] = max(state.privacy[inp] for inp in inputs) + 2.5  # Example rule
        
        # Add more ops: BN, Conv, etc.
    
    return state
