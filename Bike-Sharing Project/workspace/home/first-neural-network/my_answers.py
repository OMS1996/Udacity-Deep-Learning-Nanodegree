import numpy as np

#Here we create the architecture of the neural network.
#but without the parameters just an Empty shell of a Network.

class NeuralNetwork(object):
    
    #The Network(Self) is a 2 layered network with an input node and a learning rate as parameters.
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # Set number of nodes in input, hidden and output layers.
        ## NUMBER OF NODES.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # INITIALIZING THE WEIGHTS.
        ## numpy.random.normal(loc=0.0, scale=1.0, size=None)
        ### 0.0 , The scale , Dimensions of the weights
        
        # 1ST to 2ND layer
        # Initialize weights from [ Inputs to Hidden ] 0.0,  input_nodes**-0.5, (self.input_nodes, self.hidden_nodes)
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,(self.input_nodes, self.hidden_nodes) )
        
        # 2ND TO 3RD Layer
        # np.random.normal(0.0 , The scale , Dimensions of the weights)
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,(self.hidden_nodes, self.output_nodes))
        
        #The learning rate associated with that specific network
        self.lr = learning_rate
        

        # In Python, you can define a function with a lambda expression,
        self.activation_function = lambda x : 1/(1+np.exp(-x))
        
  
                    
    # trainning the Neural Network = FP + BP
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        
        # Rate of change for weights being initialized and declared. 
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        
        for X, y in zip(features, targets):
            
            # Forward Prop, takes in the features (every single feature.
            final_outputs, hidden_outputs = self.forward_pass_train(X) 
            
            # Back proagation, takes in outputs of Forward prop, X & y and the rate of change for all the weights
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Forward propagation for the two layered neural network.
            
            Arguments
            ---------
            X: features batch
        '''
        
        ### Forward pass ###
        
        # Hidden layer.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # Signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # Signals from hidden layer

        # Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
    
        # Note to self: reason of returning the hidden output is primairly due to the fact that we will backpropagate later.
        return final_outputs, hidden_outputs # return the final output as well as the hiddent output
    

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        
        ### Backward pass ###
        
        # PART I: getting the total error of each layer
        ## Output error
        ### Note to self: If you flip the Network upside down this becomes the input if you think about it
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        ### The Entire hidden layer's contribution to the error, The contributing error from the weights 
        hidden_error = np.dot(self.weights_hidden_to_output,error) #IMPORTANT weights multiplied by the previous error tern
        
        # Part II: Getting the error term through differentiation of the output
        ## Backpropagated ERROR TERMS (Sigma_error)
        ###The differentiation of the output terms for each of the layers
        # No wights here,
        output_error_term = error # Note to self: this is regression, so no activation function so gradient equals 1
        #            weights* error_term * diferentiation of hidden layer output    
        hidden_error_term = hidden_error * hidden_outputs*(1-hidden_outputs)
        
        # Part III:
        ## Now that we have the error, the gradient of the output and input we can finally get the the delta
        
        ### Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]

        ### Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        
        return delta_weights_i_h, delta_weights_h_o


    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += (self.lr*delta_weights_h_o)/n_records 
        # update hidden-to-output weights with gradient descent step.
        
        self.weights_input_to_hidden += (self.lr*delta_weights_i_h)/n_records 
        # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' 
            Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        ####  Forward pass 
        # Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = features 
        hidden_outputs = self.activation_function(np.dot(hidden_inputs,self.weights_input_to_hidden)) 
        
        # Output layer - 
        final_inputs = hidden_outputs
        final_outputs = np.dot(final_inputs,self.weights_hidden_to_output) 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5102
learning_rate = 0.699
hidden_nodes = 6
output_nodes = 1
