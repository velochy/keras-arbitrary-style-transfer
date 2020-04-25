import tensorflow as tf
from utils import load_json_h5_model
import sys
from tensorflow.python.framework import convert_to_constants

model = load_json_h5_model(sys.argv[1])
model.summary()

ispec = tuple(map(lambda inp: tf.TensorSpec(inp.shape, inp.dtype, name=inp.name),model.input))
print("ISPEC",ispec)
full_model = tf.function(lambda x1,x2: model([x1,x2]), input_signature=ispec)

#ispec = (tf.TensorSpec(model.input.shape, model.input.dtype),)
#print("ISPEC",ispec)

full_model = tf.function(lambda x1, x2: model([x1,x2]), input_signature=ispec)
full_model = full_model.get_concrete_function()

# Get frozen ConcreteFunction
frozen_func = convert_to_constants.convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
#for layer in layers:
#    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name=sys.argv[2],
                  as_text=False)
