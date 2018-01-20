import numpy as np
import random


def inttobin(x):
    v = "{0:b}".format(x)
    return v


# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


# Propagate
def propagate(x, y, lr, inputlayer_neurons, hiddenlayer_neurons, output_neurons, wh, bh, wout, bout):
    hidden_layer_input1 = np.dot(x, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    E = y - output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += x.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr
    return (output, wh, bh, wout, bout)


def forwardpropagate(x, inputlayer_neurons, lr, hiddenlayer_neurons, output_neurons, wh, bh, wout, bout):
    hidden_layer_input1 = np.dot(x, wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    return output


# User input variables
transformer = float(input("Transformer?"))
epoch = int(input("Epochs?"))

# Input/Output arrays, random binary integers for input, doubled for output
x = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
y = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])

for i in range(1000):
    randi = random.randint(0, 255)
    randidub = int(randi * transformer)
    randb = inttobin(randi)
    randbdub = inttobin(randidub)
    randba = np.array([])
    randbduba = np.array([])

    if (len(randb) != 8):
        for i in range(0, 8 - len(randb)):
            randb = "0" + randb

    if (len(randbdub) != 8):
        for i in range(0, 8 - len(randbdub)):
            randbdub = "0" + randbdub

    for e in range(0, 8): randba = np.append(randba, int(randb[e]))
    for e in range(0, 8): randbduba = np.append(randbduba, int(randbdub[e]))

    x = np.block([[x], [randba]])
    y = np.block([[y], [randbduba]])

# Variable initialization

lr = 0.01  # Setting learning rate
inputlayer_neurons = x.shape[1]  # number of features in data set
hiddenlayer_neurons = 10
output_neurons = 8  # number of neurons at output layer
output = None
# weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
bh = np.random.uniform(size=(1, hiddenlayer_neurons))
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

for i in range(epoch):
    # Propagation
    unpack = propagate(x, y, lr, inputlayer_neurons, hiddenlayer_neurons, output_neurons, wh, bh, wout, bout)
    output = unpack[0]
    wh = unpack[1]
    bh = unpack[2]
    wout = unpack[3]
    bout = unpack[4]

print(output)
while True:
    out = ""
    rawin = input("Binary in")
    if (rawin == "break"):
        break
    xvalue = [int(x) for x in rawin.split(",")]
    unpack = forwardpropagate(xvalue, lr, inputlayer_neurons, hiddenlayer_neurons, output_neurons, wh, bh, wout, bout)
    for x in unpack[0]:
        if (x >= .5):
            x = 1
        else:
            x = 0
        out = out + str(x)
    print(out)
