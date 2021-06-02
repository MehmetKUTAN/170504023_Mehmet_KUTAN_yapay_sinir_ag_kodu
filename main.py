import numpy as np
import matplotlib.pyplot as plt

train_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50],
                       [1, 85, 66, 29, 0, 26.6, 0.351, 31],
                       [8, 183, 64, 0, 0, 23.3, 0.672, 32],
                       [1, 89, 66, 23, 94, 28.1, 0.167, 21]])
train_data_target = np.array([[0, 1, 0, 1, 0, 1, 0, 1],
                              [1, 0, 1, 0, 1, 0, 1, 0],
                              [0, 1, 0, 1, 0, 1, 0, 1],
                              [1, 0, 1, 0, 1, 0, 1, 0]])

input_neuron_size = train_data.shape[1]
hidden_neuron_size = 3
output_neuron_size = train_data_target.shape[1]

train_data_size = train_data.shape[0]

weight1 = np.random.rand(input_neuron_size, hidden_neuron_size)
weight2 = np.random.rand(hidden_neuron_size, output_neuron_size)

eta = 0.7
epoch = 1000

error_history = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(x):
    output_inputs = x

    output_hiddens = np.zeros(hidden_neuron_size)
    for i in range(input_neuron_size):
        for j in range(hidden_neuron_size):
            output_hiddens[j] = output_hiddens[j] + output_inputs[i] * weight1[i][j]

    for j in range(hidden_neuron_size):
        output_hiddens[j] = sigmoid(output_hiddens[j])

    output_outputs = np.zeros(output_neuron_size)
    for i in range(hidden_neuron_size):
        for j in range(output_neuron_size):
            output_outputs[j] = output_outputs[j] + output_hiddens[i] * weight2[i][j]

    for j in range(hidden_neuron_size):
        output_outputs[j] = sigmoid(output_outputs[j])

    return output_outputs


for iter in range(epoch):
    total_error = 0
    for train_data_i in range(train_data_size):

        out_inputs = train_data[train_data_i, :]

        out_hiddens = np.zeros(hidden_neuron_size)
        for i in range(input_neuron_size):
            for j in range(hidden_neuron_size):
                out_hiddens[j] = out_hiddens[j] + out_inputs[i] * weight1[i][j]

        for j in range(hidden_neuron_size):
            out_hiddens[j] = sigmoid(out_hiddens[j])

        output_outputs = np.zeros(output_neuron_size)
        for i in range(hidden_neuron_size):
            for j in range(output_neuron_size):
                output_outputs[j] = output_outputs[j] + out_hiddens[i] * weight2[i][j]

        for j in range(output_neuron_size):
            output_outputs[j] = sigmoid(output_outputs[j])

        delta_outputs = np.zeros(output_neuron_size)
        for i in range(output_neuron_size):

            delta_outputs[i] = (train_data_target[train_data_i][i] - output_outputs[i]) * output_outputs[i] * (
                    1 - output_outputs[i])

            for i in range(hidden_neuron_size):
                for j in range(output_neuron_size):
                    weight2[i][j] = weight2[i][j] + eta * delta_outputs[j] * out_hiddens[i]

            delta_hiddens = np.zeros(hidden_neuron_size)
            for i in range(hidden_neuron_size):
                total_sum = 0
                for j in range(output_neuron_size):
                    total_sum = total_sum + (weight2[i][j] * delta_outputs[j])
                delta_hiddens[i] = out_hiddens[i] * (1 - out_hiddens[i]) * total_sum

            for i in range(input_neuron_size):
                for j in range(hidden_neuron_size):
                    weight1[i][j] = weight1[i][j] + eta * delta_hiddens[j] * out_inputs[i]

            total_error = total_error + np.sum(np.power(train_data_target[train_data_i] - output_outputs, 2))
        error_history.append(total_error)
        print('ıterasyon(epoch):', iter, ' hata:', total_error)

    plt.figure(figsize=(5, 5))
    plt.plot(error_history)
    plt.xlabel('Iterasyon(epoch)')
    plt.ylabel('Hata(error)')
    plt.show()
