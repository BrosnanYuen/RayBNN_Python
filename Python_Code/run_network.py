import numpy as np
import raybnn_python
import mnist
import os 
from sklearn.metrics import accuracy_score

def main():

    if os.path.isfile("./train-labels-idx1-ubyte.gz") == False:
        mnist.init()

    x_train, y_train, x_test, y_test = mnist.load()


    max_value = np.max(x_train)
    min_value = np.min(x_train)
    mean_value = (max_value+min_value)/2

    x_train = (x_train.astype(np.float32) - mean_value)/(max_value - min_value)
    x_test = (x_test.astype(np.float32) - mean_value)/(max_value - min_value)

    print(x_train)
    print(x_train.shape)

    print(y_train)
    print(y_train.shape)

    print(y_test)
    print(y_test.shape)


    dir_path = "/tmp/"

    max_input_size = 784
    input_size = 784

    max_output_size = 10
    output_size = 10

    max_neuron_size = 20000

    batch_size = 1000
    traj_size = 1

    training_samples = 60
    crossval_samples = 60
    testing_samples = 10


    train_x = np.zeros((input_size,batch_size,traj_size,training_samples)).astype(np.float32)
    train_y = np.zeros((output_size,batch_size,traj_size,training_samples)).astype(np.float32)

    for i in range(x_train.shape[0]):
        j = (i % batch_size)
        k = int(i/batch_size)

        train_x[:, j , 0, k ] = x_train[i,:]

        idx = y_train[i]
        train_y[idx , j , 0, k ] = 1.0

    crossval_x = np.copy(train_x)
    crossval_y = np.copy(train_y)



    arch_search = raybnn_python.create_start_archtecture(
        input_size,
        max_input_size,
        output_size,
        max_output_size,
        max_neuron_size,
        batch_size,
        traj_size,
        dir_path
    )


    neuron_rad = arch_search["neural_network"]["netdata"]["neuron_rad"]
    arch_search["neural_network"]["netdata"]["proc_num"] = 4


    arch_search = raybnn_python.add_neuron_to_existing3(
        15000,
		30,
		neuron_rad*60.0,
		neuron_rad*60.0,
		neuron_rad*60.0,

        arch_search,
    )

    raybnn_python.print_model_info(arch_search)


    stop_strategy = "STOP_AT_TRAIN_LOSS"
    lr_strategy = "NONE"
    lr_strategy2 = "MAX_ALPHA"

    loss_function = "sigmoid_cross_entropy_10"

    max_epoch = 10000
    stop_epoch = 10000
    stop_train_loss = 0.0001

    exit_counter_threshold = 100
    shuffle_counter_threshold = 10000



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

    test_x = np.zeros((input_size,batch_size,traj_size,testing_samples)).astype(np.float32)
 
    for i in range(x_test.shape[0]):
        j = (i % batch_size)
        k = int(i/batch_size)

        test_x[:, j , 0, k ] = x_test[i,:]


    output_y = raybnn_python.test_network(
        test_x,

        arch_search
    )

    print(output_y.shape)

    pred = []
    for i in range(x_test.shape[0]):
        j = (i % batch_size)
        k = int(i/batch_size)

        sample = output_y[:, j , 0, k ]
        print(sample)

        pred.append(np.argmax(sample))


    acc = accuracy_score(y_test, pred)

    print(acc)




if __name__ == '__main__':
    main()




