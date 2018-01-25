import numpy as np
import random
import tkinter

#integer to binary
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
    return [output, hiddenlayer_activations,wh,bh,wout,bout]

def train():
    # Variable initialization

    # Input/Output arrays, random binary integers for input, doubled for output
    x = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
    y = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])

    train.lr = 0.01  # Setting learning rate
    train.inputlayer_neurons = x.shape[1]  # number of features in data set
    train.hiddenlayer_neurons = 8
    train.output_neurons = 8  # number of neurons at output layer
    output = None
    # weight and bias initialization
    train.wh = np.random.uniform(size=(train.inputlayer_neurons, train.hiddenlayer_neurons))
    train.bh = np.random.uniform(size=(1, train.hiddenlayer_neurons))
    train.wout = np.random.uniform(size=(train.hiddenlayer_neurons, train.output_neurons))
    train.bout = np.random.uniform(size=(1, train.output_neurons))

    for i in range(clicked2.epoch):
        randi = random.randint(0, 255)
        randidub = int(randi * clicked1.transformer)
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


    for i in range(clicked2.epoch):
        # Propagation
        unpack = propagate(x, y, train.lr, train.inputlayer_neurons, train.hiddenlayer_neurons, train.output_neurons, train.wh, train.bh, train.wout, train.bout)
        output = unpack[0]
        train.wh = unpack[1]
        train.bh = unpack[2]
        train.wout = unpack[3]
        train.bout = unpack[4]
def solve():
    ltxtoutput = [0,0,0,0,0,0,0,0]
    txtoutput =""
    x = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    xstr = mainloop.app.solvetxt.get()
    for e in range(8):
        x[0, e] = xstr[e]

    unpack = forwardpropagate(x,train.inputlayer_neurons, train.lr, train.hiddenlayer_neurons, train.output_neurons, train.wh, train.bh, train.wout, train.bout)
    output = unpack[0]
    hiddenlayer_activations = unpack[1]
    for e in range(8):
        if output[0,e] >= .5:
            ltxtoutput[e] = 1
        else:
            ltxtoutput[e] = 0
    for e in range(8):
        if ltxtoutput[e] == 1:
            txtoutput = txtoutput + "1"
        else:
            txtoutput = txtoutput + "0"
    mainloop.app.solvelabel.configure(text=txtoutput)
    f1 = int(255-(output[0,0]* 255))
    f2 = int((255-output[0,1] * 255))
    f3 = int((255-output[0,2] * 255))
    f4 = int((255-output[0,3] * 255))
    f5 = int((255-output[0,4] * 255))
    f6 = int((255-output[0,5] * 255))
    f7 = int((255-output[0,6] * 255))
    f8 = int((255-output[0,7] * 255))
    mainloop.app.canvas.itemconfig(mainloop.app.foval1, fill ="#%02x%02x%02x" % (f1, f1, f1))
    mainloop.app.canvas.itemconfig(mainloop.app.foval2, fill="#%02x%02x%02x" % (f2, f2, f2))
    mainloop.app.canvas.itemconfig(mainloop.app.foval3, fill="#%02x%02x%02x" % (f3, f3, f3))
    mainloop.app.canvas.itemconfig(mainloop.app.foval4, fill="#%02x%02x%02x" % (f4, f4, f4))
    mainloop.app.canvas.itemconfig(mainloop.app.foval5, fill="#%02x%02x%02x" % (f5, f5, f5))
    mainloop.app.canvas.itemconfig(mainloop.app.foval6, fill="#%02x%02x%02x" % (f6, f6, f6))
    mainloop.app.canvas.itemconfig(mainloop.app.foval7, fill="#%02x%02x%02x" % (f7, f7, f7))
    mainloop.app.canvas.itemconfig(mainloop.app.foval8, fill="#%02x%02x%02x" % (f8, f8, f8))
    m1 = int((255-hiddenlayer_activations[0,0] * 255))
    m2 = int((255-hiddenlayer_activations[0,1] * 255))
    m3 = int((255-hiddenlayer_activations[0,2] * 255))
    m4 = int((255-hiddenlayer_activations[0,3] * 255))
    m5 = int((255-hiddenlayer_activations[0,4] * 255))
    m6 = int((255-hiddenlayer_activations[0,5] * 255))
    m7 = int((255-hiddenlayer_activations[0,6] * 255))
    m8 = int((255-hiddenlayer_activations[0,7] * 255))
    mainloop.app.canvas.itemconfig(mainloop.app.moval1, fill="#%02x%02x%02x" % (m1, m1, m1))
    mainloop.app.canvas.itemconfig(mainloop.app.moval2, fill="#%02x%02x%02x" % (m2, m2, m2))
    mainloop.app.canvas.itemconfig(mainloop.app.moval3, fill="#%02x%02x%02x" % (m3, m3, m3))
    mainloop.app.canvas.itemconfig(mainloop.app.moval4, fill="#%02x%02x%02x" % (m4, m4, m4))
    mainloop.app.canvas.itemconfig(mainloop.app.moval5, fill="#%02x%02x%02x" % (m5, m5, m5))
    mainloop.app.canvas.itemconfig(mainloop.app.moval6, fill="#%02x%02x%02x" % (m6, m6, m6))
    mainloop.app.canvas.itemconfig(mainloop.app.moval7, fill="#%02x%02x%02x" % (m7, m7, m7))
    mainloop.app.canvas.itemconfig(mainloop.app.moval8, fill="#%02x%02x%02x" % (m8, m8, m8))
    o1 = (255-(x[0,0]) * 255)
    o2 = (255-(x[0,1]) * 255)
    o3 = (255-(x[0,2]) * 255)
    o4 = (255-(x[0,3]) * 255)
    o5 = (255-(x[0,4]) * 255)
    o6 = (255-(x[0,5]) * 255)
    o7 = (255-(x[0,6]) * 255)
    o8 = (255-(x[0,7]) * 255)
    mainloop.app.canvas.itemconfig(mainloop.app.ooval1, fill="#%02x%02x%02x" % (o1, o1, o1))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval2, fill="#%02x%02x%02x" % (o2, o2, o2))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval3, fill="#%02x%02x%02x" % (o3, o3, o3))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval4, fill="#%02x%02x%02x" % (o4, o4, o4))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval5, fill="#%02x%02x%02x" % (o5, o5, o5))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval6, fill="#%02x%02x%02x" % (o6, o6, o6))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval7, fill="#%02x%02x%02x" % (o7, o7, o7))
    mainloop.app.canvas.itemconfig(mainloop.app.ooval8, fill="#%02x%02x%02x" % (o8, o8, o8))
    for e in range(8):
        for i in range(8):
            c = unpack[2][e][i]
            c = 255 - (c * 255 * x[0,e])
            c = int(c + unpack[3][0][e])
            if c <= 0:
                c=0
            mainloop.app.canvas.create_line(260,(e*70)+70,650,(i*70)+70,fill="#%02x%02x%02x" % (c, c, c))
    for e in range(8):
        for i in range(8):

            c = unpack[4][e][i]
            c = 255 - (c * 255 * hiddenlayer_activations[0,e])
            c=int(c+unpack[5][0][e])

            if c <= 0:
                c=0
            mainloop.app.canvas.create_line(650,(e*70)+70,1020,(i*70)+70,fill="#%02x%02x%02x" % (c, c, c))




#GUI
def clicked1():
    clicked1.transformer = int(mainloop.app.transformertxt.get())

def clicked2():
    clicked2.epoch = int(mainloop.app.epochtxt.get())

class Base(tkinter.Frame):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        o1 = 255
        o2 = 255
        o3 = 255
        o4 = 255
        o5 = 255
        o6 = 255
        o7 = 255
        o8 = 255
        self.master.title("nn-binary")
        self.pack(fill="both", expand=1)
        transformerlabel = tkinter.Label(self, text="Transformer")
        transformerlabel.grid(column=0, row =0)
        self.transformertxt = tkinter.Entry(self, width=2)
        self.transformertxt.grid(column=0, row=1)
        transformerbtn = tkinter.Button(self, text="Enter", command=clicked1)
        transformerbtn.grid(column=0, row=2)
        solvebtn = tkinter.Button(self, text="Solve", command=solve)
        solvebtn.grid(column=16, row=1)
        self.solvelabel = tkinter.Label(self)
        self.solvelabel.grid(column=18, row=1)
        epochlabel = tkinter.Label(self, text="Epoch")
        epochlabel.grid(column=1, row=0)
        self.epochtxt = tkinter.Entry(self, width=5)
        self.epochtxt.grid(column=1, row=1)
        self.solvetxt = tkinter.Entry(self, width=10)
        self.solvetxt.grid(column=15, row=1)
        epochbtn = tkinter.Button(self, text="Enter", command=clicked2)
        epochbtn.grid(column=1, row=2)
        trainbtn = tkinter.Button(self, text="Train", command=train)
        trainbtn.grid(column=2, row =1)
        self.canvas = tkinter.Canvas(self, width=1280, height=640)
        self.canvas.place(x=0, y=80)
        self.ooval1 = self.canvas.create_oval(240, 50, 280, 90, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval2 = self.canvas.create_oval(240, 120, 280, 160, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval3 = self.canvas.create_oval(240, 190, 280, 230, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval4 = self.canvas.create_oval(240, 260, 280, 300, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval5 = self.canvas.create_oval(240, 330, 280, 370, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval6 = self.canvas.create_oval(240, 400, 280, 440, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval7 = self.canvas.create_oval(240, 470, 280, 510, fill="#%02x%02x%02x" % (255,255,255))
        self.ooval8 = self.canvas.create_oval(240, 540, 280, 580, fill="#%02x%02x%02x" % (255,255,255))
        fx1= 630
        fx2 = fx1+40
        self.moval1 = self.canvas.create_oval(fx1, 50, fx2, 90, fill="#%02x%02x%02x" % (255,255,255))
        self.moval2 = self.canvas.create_oval(fx1, 120, fx2, 160, fill="#%02x%02x%02x" % (255, 255, 255))
        self.moval3 = self.canvas.create_oval(fx1, 190, fx2, 230, fill="#%02x%02x%02x" % (255, 255, 255))
        self.moval4 = self.canvas.create_oval(fx1, 260, fx2, 300, fill="#%02x%02x%02x" % (255, 255, 255))
        self.moval5 = self.canvas.create_oval(fx1, 330, fx2, 370, fill="#%02x%02x%02x" % (255, 255, 255))
        self.moval6 = self.canvas.create_oval(fx1, 400, fx2, 440, fill="#%02x%02x%02x" % (255, 255, 255))
        self.moval7 = self.canvas.create_oval(fx1, 470, fx2, 510, fill="#%02x%02x%02x" % (255, 255, 255))
        self.moval8 = self.canvas.create_oval(fx1, 540, fx2, 580, fill="#%02x%02x%02x" % (255, 255, 255))
        mx1=1000
        mx2=mx1+40
        self.foval1 = self.canvas.create_oval(mx1, 50, mx2, 90, fill="#%02x%02x%02x" % (o1, o1, o1))
        self.foval2 = self.canvas.create_oval(mx1, 120, mx2, 160, fill="#%02x%02x%02x" % (o2, o2, o2))
        self.foval3 = self.canvas.create_oval(mx1, 190, mx2, 230, fill="#%02x%02x%02x" % (o3, o3, o3))
        self.foval4 = self.canvas.create_oval(mx1, 260, mx2, 300, fill="#%02x%02x%02x" % (o4, o4, o4))
        self.foval5 = self.canvas.create_oval(mx1, 330, mx2, 370, fill="#%02x%02x%02x" % (o5, o5, o5))
        self.foval6 = self.canvas.create_oval(mx1, 400, mx2, 440, fill="#%02x%02x%02x" % (o6, o6, o6))
        self.foval7 = self.canvas.create_oval(mx1, 470, mx2, 510, fill="#%02x%02x%02x" % (o7, o7, o7))
        self.foval8 = self.canvas.create_oval(mx1, 540, mx2, 580, fill="#%02x%02x%02x" % (o8, o8, o8))

def mainloop():
    root= tkinter.Tk()
    root.geometry("1280x720+400+400")
    mainloop.app = Base()
    root.mainloop()


mainloop()