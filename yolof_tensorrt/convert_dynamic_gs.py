import onnx_graphsurgeon as gs
import onnx
import numpy as np

graph = gs.import_onnx(onnx.load("./test3.onnx"))

for inp in graph.inputs:
    inp.shape[0] = gs.Tensor.DYNAMIC
for out in graph.outputs:
    out.shape[0] = gs.Tensor.DYNAMIC

# Remove the non-used node from the graph completely
graph.cleanup()

onnx.save(gs.export_onnx(graph), "./model_gs.onnx")

