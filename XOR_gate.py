import numpy as np
#We have to train the data
    #Creating a vector containing the logic of the xor gate
possible_inputs=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
possible_value_out=np.array([[0],[1],[1],[0]],dtype=float)



#Creating the Neural Network itself
input_neurons=2 #We have 2
hidden_neurons=4
output_neurons=1 #The gate give us one solution
#For the 2 inputs we create the weights and the biases
#In the begining we give them random numbers and then during training the neural
#network adjusts them to impove its predictions
#In every weight we give a matrix of random numbers 2x4 grid containing 8 random numbers
#We have 8 different numbers beacuse each neuron has its own weight
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.5
#For the bias we dont need random numbers and we create a 0 bias
b1 = np.zeros((1, hidden_neurons))
#this weight because its connected with the output is 4x1 containing 4 random numbers
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.5
b2 = np.zeros((1, output_neurons))


#Activation
#With this fuction we give the result a value between (0,1)
def sigmoid(z):
 return 1 / (1 + np.exp(-z))


#Trainng the Machine
#Learning rate is the percentage of the error correction , how much we let the system change in one session
learning_rate = 1.0
#How many session we give the machine to learn
epochs = 20000
#m Gives the sum of the training examples we have to give to the machine to train
m = len(possible_inputs)

for epoch in range(epochs):

    # ----- Forward propagation -----
    #We caclulate the fuction of the first layer on the machine
    z1 = possible_inputs @ W1 + b1
    a1 = sigmoid(z1)
    #The second layer of the machine
    z2 = a1 @ W2 + b2
    #We make the values of the machine between 0 and 1
    a2 = sigmoid(z2)

    # ----- Binary cross-entropy loss -----
    #After use compute the output weight that the network give us
    #We see compare the number with the number of the output we expect
    eps = 1e-9 #We use this because we cant use the log(0)
    #We see in the loss fuction we have 2 parts 1 with the Binary Cross entropy it gives a clear 0 or 1
    #The first half is only used when the correct answer is 1 and the second when the correct answer is 0.
    loss = -np.mean(possible_value_out* np.log(a2 + eps) + (1 - possible_value_out) * np.log(1 - a2 + eps))

    # ----- Backpropagation -----
    # Output layer gradients
    #After we see how much we correct we are of how far we are from the actual values
    #We tune the weights to get a better aproximation
    #First we go from the desize output layer to the first hidden and we tune the weighths and the bias
    dz2 = (a2 - possible_value_out) / m
    #We take the output and multiply with the weights to see with of the neurons effect the most the output we want
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    #Then we go from the 2 layer to the frist hidden layer that is connected with the input
    # Hidden layer gradients
    #Before we update the fist set of weights we need to see how much to blame the hidden layers error.And
    #the neuron that has a huge weight will be the one we change the most
    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
    #We know exaclty what cahnges we need to do so we fix the hidden weights
    dW1 =possible_inputs.T @ dz1
    #Every example produces it own signal and to determine what we need to cahnge we have to
    #sum up all the individual errors from every example in the batch
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ----- Gradient descent
    #After we have done all the work on how the system behaves in a single training sesion
    #Now we have to aplly this changes in the system and the weights of the fuction
    #This is the moment that we tune the system
    #We can say that the weight is the slope of the line and the bias is the offset
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    #After ther training we get the statistics for our network .How many times we trained and how accurate we are
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")



# ==========================================
# 5. TEST THE TRAINED NETWORK
# ==========================================
print("\nFinal predictions:")
#We ran for the last time the machine to see the outputs it gives back to us
z1 = possible_inputs @ W1 + b1
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2
predictions = sigmoid(z2)

for inputs, prediction, target in zip(possible_inputs, predictions, possible_value_out):
    result = int(prediction[0] >= 0.5)

    print(
        f"Input: {inputs.astype(int)} | "
        f"Prediction: {prediction[0]:.4f} | "
        f"Class: {result} | "
        f"Target: {int(target[0])}"
    )