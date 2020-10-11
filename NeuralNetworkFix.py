#Ofir Shimshon
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse(output_layer_outputs, all_y_trues, N):
  # y_true and y_pred are numpy arrays of the same length.
  # mse = ((output_layer_outputs - y)**2).sum() / (2*N)
  return ((output_layer_outputs - all_y_trues) **2).sum() / (2*N)

# DATASET
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

#TRUE PREDICTIONS
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])
N=all_y_trues.size

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons
    - an output layer with 1 neuron
  '''
  
  def __init__(self):
    np.random.seed(10)
    n_input = 2  #Inputs
    n_hidden = 2 #Hidden Layer Neurons
    n_output = 1 #Outputs
    self.error_history = []
    self.epoch_list = []    
    self.learning_rate = 0.1

    
    self.weights_1 = np.random.normal(scale=0.5, size=(n_input, n_hidden)) #(2,2) Input Weights
    self.weights_2 = np.random.normal(scale=0.5, size=(n_hidden,n_output))# (2,1) Hidden Layer Weights
    self.biases1 = np.random.normal(scale=0.5, size=(1,n_hidden))
    self.biases2 = np.random.normal(scale=0.5, size=(1,n_output))
    
  def feedforward_And_BackPropagation(self, x):
    self.hidden_layer_inputs =np.add(np.dot(x,self.weights_1), self.biases1)
    self.hidden_layer_outputs = sigmoid(self.hidden_layer_inputs)
    self.output_layer_inputs = np.add(np.dot(self.hidden_layer_outputs, self.weights_2),self.biases2)
    self.output_layer_outputs = sigmoid(self.output_layer_inputs)
    #BackProp
    self.backpropagation(x,self.hidden_layer_outputs,self.output_layer_outputs)
    return self.output_layer_outputs
    
    
  def backpropagation(self,x,hidden_layer_output, output_layer_output):
      output_layer_error = output_layer_output - all_y_trues
      output_layer_delta = output_layer_error * deriv_sigmoid(output_layer_output)
      hidden_layer_error = np.dot(output_layer_delta.T, self.weights_2.T)
      hidden_layer_delta = hidden_layer_error *  deriv_sigmoid(hidden_layer_output)
      
      #Weight updates
      weights2_update = np.dot(hidden_layer_output.T, output_layer_delta) / N
      weights1_update = np.dot(x.T, hidden_layer_delta.T) / N
      
      self.biases1-= self.learning_rate * np.sum(hidden_layer_delta)
      self.biases2-= self.learning_rate * np.sum(output_layer_delta)
      self.weights_2 -= self.learning_rate * weights2_update
      self.weights_1 -= self.learning_rate * weights1_update
      
      
  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- FeedForward And BackPropagation
        output_layer_output = self.feedforward_And_BackPropagation(x)
        #Save error history for error graph
        self.error_history.append(np.average(np.abs(self.error)))
        self.epoch_list.append(epoch)
      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        loss = mse(output_layer_output,all_y_trues, N)
        print("Epoch %d loss: %.3f" % (epoch, loss))



# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues) # Train the Network with Data, The Answers and learning rate = 0.1

#plot the error throughout the training
plt.figure(figsize=(15,5))
plt.plot(network.epoch_list, network.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
