# Radial-Basis-Function Network

[Radial-Basis-Function Network](https://en.wikipedia.org/wiki/Radial_basis_function_network) using [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to adjust RBF-centers, RBF-widths and weights.
Implemented in python, for a programming assignment of course ***Technical Neuroal Network***.
 
# Using Instruction

Simply import function "RBFNetwork" from RBFNetwork.py

	$ python
	from RBFNetwork import RBFNetwork
	
Now you can use this function with the following definition:

	Errors, TNN = RBFNetwork (N, M, K, Weights, Patterns, LearningRateC, LearningRateS, LearningRateW, RandomSeed, MaxSteps, Batch)

### N, M, K

Number of inputs, outputs and RBFs.

### Weights: Initial weights

Weights: an array consist of M arrays, for each output neuron, in which there are k numbers for weights of each RBF neuron, 
**Alternatively**, direction of a .dat file in which you must put weights with the aforementioned order (lines with # consider as a comment), 
**Alternatively**, '' or [] for random initialization with vlues between -0.5 and 0.5

### Patterns

A list of P patterns used to train the perzeptron in the form of tuples in which the first element is a list containing N inputs and the second one is a list containing M outputs.
**Alternatively**, source direction of  a .dat file in which for each training pattern you must put the inputs followed by outputs followed by next patterns (lines with # consider as a comment).

### LearningRateC, LearningRateS, LearningRateW

Learning rate dor adjusting Centers, Widths and Weights.

### RandomSeed

A random seed used for random initializing and shuffling, to be able to reproduce results.

### MaxSteps

The maximum number of iterations in which the model can train.

### Batch

A boolean value which is 1 for batch learning and 0 for single-step learning

### Errors

A list containing the squared error of each pattern in each of the iterations.

### TNN

A function which is a MLP model with calculated weights, with the following definition:

	Y = TNN(X)
	
**X**: a list of containing N values as input
**Y**: a list containing M values as output

# Example:

*Please check "RBFNetwork-Test.py" for an example.*

*To plot the learning curve of the model using gnuplot, a function called gnuplotOut which makes a file readable for gnuplot had been implemented. Check the example for more details.*

![Check Train.plt to see training patterns used to train example model](/Training_Patterns.png)

![Check Test.plt to see result of using example model on validation dataset](/Test.png)

![Check Learning_Curve.plt to see learning curver of example model](/Learning_Curve.png)

# Authors 

	Ali Mohammadi
	Rozhin Bayati


*Best Regards*