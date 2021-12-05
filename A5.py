import numpy
import numpy as np
import matplotlib.pyplot as plt 

# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])


class NeuralNetwork:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    def sigmoid(self, x, deriv=False):
        # Task 1 - 20 points

        # Implement the sigmoid activation function
        # if deriv = True, return the derivative of the the sigmoid function der_sig
        # else calculate the sigmoid function and return sig
        # Hint: derivate of sig = x * (1-x)
        # activation function ==> Sig = 1/1+e^(-x)

        # Your code here
        if deriv:
            return x * (1-x)
        else:
            return 1/(1+pow(numpy.e,-x))
        # return sig and der_sig accordingly

    def feed_forward(self):
        # Task 2 - 10 points

        # Implement feed forward to allow the data to flow through the neural network
        # The result of this feed-forward function will be the output of the hidden layer
        # You have to write the code that will set the output of the hidden layers

        # Multiply the inputs and the corresponding weights
        inputs_dot_weights = numpy.dot(self.inputs,self.weights)
        
        # Call the sigmoid function() and pass inputs_dot_weights as parameter
        self.hidden = self.sigmoid(inputs_dot_weights)

    def backpropagation(self):
        # Backpropagation will go back through the layer(s) of the neural network,
        # determine which weights contributed to the output and the error,
        # then change the weights based on the gradient of the hidden layers output.

        # Task 3 - 10 points
        
        # Step 1: Find error : Error will be the difference between the correct output matrix y
        # and hidden layers output got from feed_forward

        self.error = self.outputs - self.hidden

        # Step 2: Calculates delta which is nothing but the multiplication of error and the derivative of the
        # hidden layers prediction
        delta = self.error * self.sigmoid(self.hidden, deriv=True)

        # Step 3: Change weights based on gradient of hidden layers
        # weights = weights +  transpose of input multiplied with delta
        self.weights += numpy.dot(numpy.transpose(self.inputs), delta)

    # Now we will train the neural net for 25,000 iterations
    #epochs = 25000
    def train(self, epochs=25000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            
            # Task 4 - 10 points

            # Calculate the average of the absolute error and add the result to error_history,
            # Complete the below line (You only have to write code within the append())
            self.error_history.append(numpy.average(abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction


# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

# create two new test examples to predict                                   
test_1 = np.array([[1, 1, 0]])
test_2 = np.array([[0, 1, 1]])

# print the predictions for both examples                                   
print(NN.predict(test_1), ' - Correct: ', test_1[0][0])
print(NN.predict(test_2), ' - Correct: ', test_2[0][0])

# Plot Epoch vs error plot to understand error rate for different epochs
# X axis = Epochs, Y= Error
# Label the plot accordingly
plt.figure(figsize=(15, 5))

plt.xlabel = "Epochs"
plt.ylabel = "Error"

plt.plot(NN.epoch_list, NN.error_history)
plt.show()
