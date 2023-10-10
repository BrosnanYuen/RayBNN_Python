import numpy as np
import raybnn_python




def main():
    dir_path = "/tmp/"

    max_input_size = 162
    input_size = 162

    max_output_size = 2
    output_size = 2

    max_neuron_size = 600

    batch_size = 100
    traj_size = 20

    training_samples = 50
    crossval_samples = 70


    arch_search = raybnn_python.create_start_archtecture(input_size,
                                                        max_input_size,
                                                        output_size,
                                                        max_output_size,
                                                        max_neuron_size,
                                                        batch_size,
                                                        traj_size,
                                                        dir_path)


    raybnn_python.print_model_info(arch_search)


    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "COSINE_ANNEALING"
    lr_strategy2 = "BTLS_ALPHA"

    loss_function = "MAE"

    max_epoch = 10000
    stop_epoch = 10000
    stop_train_loss = 0.001

    exit_counter_threshold = 5
    shuffle_counter_threshold = 10000

    train_x = np.random.rand(input_size,batch_size,traj_size,training_samples).astype(np.float32)
    train_y = np.random.rand(output_size,batch_size,traj_size,training_samples).astype(np.float32)

    crossval_x = np.random.rand(input_size,batch_size,traj_size,crossval_samples).astype(np.float32)
    crossval_y = np.random.rand(output_size,batch_size,traj_size,crossval_samples).astype(np.float32)


    arch_search = raybnn_python.train_network(
		train_x,
        train_y,

		crossval_x,
        crossval_y,

		stop_strategy,
		lr_strategy,
		lr_strategy2,

		loss_function,
	
		max_epoch,
		stop_epoch,
		stop_train_loss,
	
		exit_counter_threshold,
		shuffle_counter_threshold,

		arch_search
    )


if __name__ == '__main__':
    main()




