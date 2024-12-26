import pydot

graph = pydot.Dot(graph_type='digraph', rankdir='LR')

input_layer = pydot.Node("Input (784)", shape="box", style="filled", fillcolor="lightblue")
maxpool1 = pydot.Node("Max-Pool1\nDepth:8\nH:128\nW:128\nFilter:8x8", shape="box", style="filled", fillcolor="lightgreen")
conv1 = pydot.Node("Conv1\nDepth:8\nH:64\nW:64\nFilter:16x16", shape="box", style="filled", fillcolor="orange")
maxpool2 = pydot.Node("Max-Pool2\nDepth:24\nH:48\nW:48\nFilter:8x8", shape="box", style="filled", fillcolor="lightgreen")
dense1 = pydot.Node("Dense1\nDepth:24\nH:16\nW:16\nFilter:8x8", shape="box", style="filled", fillcolor="lightcoral")
vector_length = pydot.Node("VectorLength\n256 | 128", shape="box", style="filled", fillcolor="yellow")

graph.add_node(input_layer)
graph.add_node(maxpool1)
graph.add_node(conv1)
graph.add_node(maxpool2)
graph.add_node(dense1)
graph.add_node(vector_length)

graph.add_edge(pydot.Edge(input_layer, maxpool1))
graph.add_edge(pydot.Edge(maxpool1, conv1))
graph.add_edge(pydot.Edge(conv1, maxpool2))
graph.add_edge(pydot.Edge(maxpool2, dense1))
graph.add_edge(pydot.Edge(dense1, vector_length))

graph.write_png('lenet_style_nn.png')
