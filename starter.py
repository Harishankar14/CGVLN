#Bunch of important imports first for the installation/ dependancy check.
import onnx 
import networkx as nx 
'''
blahh just loading the onnx model. Reffering the official onnx gr for the ref for Resnet50(for now)
I mean i would prefer something advance where the Trie coloring would be simple (TODO) 
''' 
model=onnx.load("the required model.onnx") #for now keeping the random, (trial-> resnet50) 
#just verifying the nodes within and ofc the input 
print(f"Nodes:{len(graph.nodes}})
print(f"Inputs:{graph.inputs}) #I guess i should manage the way i have to print for a specific kernal version #vague idea. 
