# Neural-Network-with-Numpy
[NeuralNetwork.py](https://github.com/Shreeyash-iitr/Neural-Network-with-Numpy/blob/master/NeuralNetwork.py) contains the code for Neural Network written from scratch with Numpy library. I have made following functions:</br>
* NN_architecture - To define architecture of neural network, in which width, depth, activation etc can be varied according to the need.
* init_parameters - To initialize parameters *RANDOMLY* or with *GLOROT* initialization.  
* Activation functions  - I have written code for 4 activations namely : sigmoid, relu, softmax, tanh. code also includes their derivative.
* full_forward_propagation - This is for forward propagation of Neural Network. This function comprise a single layer forward pass funtion.
* loss and accuracy - these functions returns cost function and accuracy after neural network is trained.
* backpropagation - This function is to apply backpropagation and compute gradients, which will be used in updating parameters in *update_parameters* function. </br>
Finally *train* and *predict* are for feeding input to the network and letting it train, and to predict target values.

![](https://github.com/Shreeyash-iitr/Neural-Network-with-Numpy/blob/master/nn.gif)

[run.py](https://github.com/Shreeyash-iitr/Neural-Network-with-Numpy/blob/master/run.py) can be used to feed data and train neural network.
