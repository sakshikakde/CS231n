Thank you Stanford University for providing all the course resources online.   
The course website: http://cs231n.stanford.edu/
# Google colab with github
Refer the wiki page of this repo.(https://github.com/sakshikakde/CS231n-Convolutional-Neural-Networks-for-Visual-Recognition-Assignments/wiki/Tools) 

# CS231n brief assignment structure

## Assignment 1
### KNN
1. Compute distance using 2 loops
2. Compute distance using 1 loops
3. Compute distance using no loops
4. Choose best value of K

### SVM
1. Compute SVM loss : naive way
2. Compute SVM loss : vectorized way
3. Implement SGD
4. Tune regularization strength and learning rate
5. Visualize the learned weights for each class


### Softmax Classifier
1. Compute softmax loss : naive way
2. Compute softmax loss : vectorized way
3. Compute gradient
4. Tune regularization strength and learning rate
5. Visualize the learned weights for each class


### Two Layer neural network
1. Implement forward pass using the weights and biases
2. Compute loss
3. Implement backpass
4. Implement train function using SGD
5. Implement predict function
6. Tune hidden layer dimension, regularization strength and learning rate
7. Visualize the learned weights for each class




## Assignment 2

### Fully-connected Neural Network
1. Implement affine layer: forward and backward
2. Implement ReLU ctivation: forward and backward
3. Sandwich layer( Affine +  ReLU): forward and backward
4. Loss layers: Softmax and SVM
5. Two layer network to get atleast 50 % accuracy
6. Fully-connected network with an arbitrary number of hidden layers.
7. Implement fancy update rules: SGD+Momentum, RMSProp and Adam

### Batch Normalization
1. Implement Batch Normalization: forward and backward
2. Fully Connected Nets with batch normalization
3. Relation between batch normalization and weight initialization
4. Relation between batch normalization and batch size
5. Implement layer normalization: forward and backward
6. Relation between layer normalization and batch size

### Dropout 
1. Implement Dropout: forward and backward
2. Fully-connected nets with Dropout
3. Comaparision of output with and without dropout

### Convolutional Networks 
1. Implement naive convolution: forward and backward
2. Implement naive max pooling: forward and backward
3. Pre implemented sandwich layers
4. Implement a three-layer ConvNet: conv - relu - 2x2 max pool - affine - relu - affine - softmax
5. Visualize Filters(learned kernals)
6. Impement spatial batch normalization: forward and backward
7. Impement group batch normalization: forward and backward

###  PyTorch on CIFAR-10 
1. Pytorch basic tutorial by Justin Johnson: https://github.com/jcjohnson/pytorch-examples
2. Barebones PyTorch: Abstraction level 1
3. PyTorch Module API: Abstraction level 2 using nn.Module
4. PyTorch Sequential API: Abstraction level 3 using nn.Sequential
5. CIFAR-10 open-ended challenge:     
My model:(conv->spatial batchnorm->relu->droupout)x3 -> maxpooling -> (affine->batchnorm->relu)x2 -> affine -> scores -> nesterov momentum         
traing accuracy:99 %, validation accuracy: 73.2 %, test accuracy = 73.5 %

## Assignment 3

### Image Captioning with RNNs

1. Download and load Microsoft COCO datset
2. Vanilla RNN: step forward, step backward
3. Vanilla RNN: forward, backward
4. Word embedding: forward, backward
5. Temporal Affine layer, Temporal Softmax loss
6. Implement forward and backward pass for the model 
7. Check model
8. Overfit RNN captioning model
9. RNN test-time sampling

### Image Captioning with LSTM

1. Download and load Microsoft COCO datset
2. LSTM: step forward, step backward
3. LSTM: forward, backward
4. Check model
5. Overfit LSTM captioning model
6. LSTM test-time sampling

### Network Visualization (PyTorch)
1. Saliency Maps
2. Fooling Images
3. Class visualization: review


