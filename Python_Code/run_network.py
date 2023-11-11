import numpy as np
import raybnn_python
import mnist
import os 


def main():

    if os.path.isfile("./train-labels-idx1-ubyte.gz") == False:
        mnist.init()

    x_train, y_train, x_test, y_test = mnist.load()

    print(x_train)
    print(x_train.shape)

    print(y_train)
    print(y_train.shape)
    return


    dir_path = "/tmp/"

    max_input_size = 784
    input_size = 784

    max_output_size = 10
    output_size = 10

    max_neuron_size = 5000

    batch_size = 100
    traj_size = 20

    training_samples = 50
    crossval_samples = 70
    testing_samples = 30


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

    loss_function = "MSE"

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

    test_x = np.random.rand(input_size,batch_size,traj_size,testing_samples).astype(np.float32)
 
    output_y = raybnn_python.test_network(
        test_x,

        arch_search
    )

    print(output_y)



if __name__ == '__main__':
    main()




