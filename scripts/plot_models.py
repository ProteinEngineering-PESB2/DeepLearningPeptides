from architectures import *
from keras.utils import plot_model

input_shape_1d = (1024,)
input_shape_2d = (1024, 150)

model = create_fully_connected(input_shape_1d, "binary")
plot_model(model, to_file = "./plots/fully_connected.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
architectures = ["conv1d", "lstm", "gru", "bilstm", "bigru", "conv1d_lstm",
    "conv1d_bilstm", "conv1d_gru", "conv1d_bigru", "conv1d_lstm_parallel",
    "conv1d_bilstm_parallel", "conv1d_gru_parallel", "conv1d_bigru_parallel"]
for arch in architectures:
    model = eval(f'create_{arch}(({input_shape_2d[0], input_shape_2d[1]}), "binary")')
    plot_model(model, to_file = f"./plots/{arch}.png", show_shapes=True, show_layer_names=True, show_layer_activations=True)
