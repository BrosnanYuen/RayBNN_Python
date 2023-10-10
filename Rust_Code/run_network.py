import numpy as np
import raybnn_python





dir_path = "/tmp/"

max_input_size = 162
input_size = 162

max_output_size = 2
output_size = 2

max_neuron_size = 600

batch_size = 100
traj_size = 20


arch_search = raybnn_python.create_start_archtecture(input_size,
max_input_size,
output_size,
max_output_size,
max_neuron_size,
batch_size,
traj_size,
dir_path)


raybnn_python.print_model_info(arch_search)








