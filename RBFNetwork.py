import random as rand
import math

def gnuplotOut(Dir, X, Y, xlabel, ylabel, title, Marker): #a simple interface for gnuplot
    file = open(Dir, 'w')
    file.write('set title'+ '"' + title + '"\n')
    file.write('set xlabel'+ '"' + xlabel + '"\n')
    file.write('set ylabel'+ '"' + ylabel + '"\n')
    file.write('plot [' + str(min(X)) + ':' + str(max(X)) + '] [' + str(min(Y)) + ':' + str(max(Y)) + '] '"'-'"' with ' + Marker + '\n')
    file.write('# x\ty\n')
    for x, y in zip(X, Y):
        file.write(str(x) + ' ' + str(y) + '\n')
    file.write ('e')
    file.close ()

def addList(W1, W2, Sub):  # pairwise sum/sub of two lists
    if not W1:
        return W2
    elif not W2:
        return W1
    elif type(W1) == list:
        return [addList(a, b, Sub) for a, b in zip(W1, W2)]
    else:
        if not Sub:
            return W1 + W2
        else:
            return W1 - W2

def readNums(File): # to read numbers only from a file for weights and patterns initialization
    f = open(File, 'r')
    data = []
    for l in f:
        l = l[0:l.find('#')] # anything followed by a '#' will be ignored
        data = data + [float(x) for x in l.split()] # two numbers must be seprated with a whitespace
    f.close()
    return data

def readPatterns(File, N, M): # to read training and test patterns from a file using readNums function and form them in a desirable way for further process 
    data = readNums(File)
    P = int(len(data)/(N+M))
    return [([data[(p * (M + N)) + n] for n in range(N)], [data[(p * (M + N)) + N + m] for m in range(M)]) for p in range(P)]
    # patterns array, contains P tuples consist of 2 arrays, the first array is input and the secod one is desired output
    # eg. [([1, 1], [1])] is a pattern array with one pattern, in which [1, 1] is input and [1] is desired output
    
def MPNeuron (X, W, F): # A simple Mcculloch&Pitch Neuron which computes net (= weighted suum of inputs) and pass it to an activation function
    return [F(sum([x * w for x, w in zip(X, ws)])) for ws in W]

def RBFNetwork (N, M, K, Weights, Patterns, LearningRateC, LearningRateS, LearningRateW, RandomSeed, MaxSteps, Batch): # Please read the "Read Me" file
    rand.seed(RandomSeed)
    # three types of initializing the weights of NN: 1. by passing it to function in the following form:
    ''' an array consist of M arrays, for each output neuron,
     in which there are k numbers for weights of each RBF neuron'''
    if not Weights: # 2. randomly initializing it using an arbitrary seed (Weights = '' or [])
        Weights = [[rand.random() - 0.5 for k in range(K)] for m in range(M)]
    elif type(Weights) == str: # 3. from a file in which weights has been saved in the above-mentioned structure
        data = readNums(Weights)
        Weights = [[data[m * K + k] for k in range(K)] for m in range(M)]
    # two types of passing training patterns, 1. direct input 2. from a file, it's similar to weights but in a structure as explained within readPatterns function
    if type(Patterns) == str:
        Patterns = readPatterns(Patterns, N, M)
    P = len(Patterns)
    rand.shuffle(Patterns)
    Centers = [] # an array consist of K arrays for each RBFunction containing N real nums (coordinates of center)
    Sizes = [] # a list containing the radial for each RBF and randomly initialized
    for i in range(K):
        Centers = Centers + [Patterns[i % P][1]] # Randomly (Shuffle) picked K patterns as Centers
        Sizes = Sizes + [rand.random()]
    ### Initialization ###
    def RBFNet (X, C, S, W):
        # a function to run a RBF neuroal network for an input using calculated centers to find euclidean distance between input and centers and pass it to gaussian function and calculate
        # their weighted sum to pass it to identity function and return output for RBF and output layers
        def EqDis(A, B):  # return the euclidean distance between A and B
            return sum([(a - b) ** 2 for a, b in zip(A, B)])
        def GaussFunction(x, s):  # the gaussian function
            return math.exp(-1 * (x ** 2) / (2 * s ** 2))
        def IdentityFunction(A):
            return A
        R = [GaussFunction(EqDis(X, c), s) for c, s in zip(C, S)] # RBF layer outputs
        Y = [IdentityFunction(sum([w * r for w, r in zip(Wm, R)])) for Wm in W] # Outputs
        return R, Y
    Error = []
    RBFNN = lambda X: RBFNet(X, Centers, Sizes, Weights)
    for step in range(MaxSteps): # updating weights, centers and sizes using learning rules based on gradient decent
        Err = []
        dW = []
        dC = []
        dS = []
        for p in Patterns:
            Xp = p[0]
            Yp = p[1]
            Rp, Outp = RBFNN(Xp)
            Err = Err + [0.5 * sum([(a - b) ** 2 for a, b in zip(Yp, Outp)])]
            dW = [[LearningRateW * (Ypm - Outpm) * r for r in Rp] for Ypm, Outpm in zip(Yp, Outp)] # learning rule for weights
            dC = [[LearningRateC * r * ((x - c) / (s ** 2)) * sum([Wm[k] * (Ypm - Outpm) for Wm, Ypm, Outpm in zip(Weights, Yp, Outp)]) for x, c in zip(Xp, Ck)] for r, Ck, s, k in zip(Rp, Centers, Sizes, range(K))] # learning rule for centers
            dS = [sum([LearningRateS * r * (((x - c) ** 2) / (s ** 3)) * sum([Wm[k] * (Ypm - Outpm) for Wm, Ypm, Outpm in zip(Weights, Yp, Outp)]) for x, c in zip(Xp, Ck)]) for r, Ck, s, k in zip(Rp, Centers, Sizes, range(K))] # learning rule for sizes
            if not Batch: # Single step mode
                Weights = addList(Weights, dW, 0)
                Centers = addList(Centers, dC, 0)
                Sizes = addList(Sizes, dS, 0)
                dW = []
                dC = []
                dS = []
                RBFNN = lambda X: RBFNet(X, Centers, Sizes, Weights)
        if Batch: # Cumulative mode
            Weights = addList(Weights, dW, 0)
            Centers = addList(Centers, dC, 0)
            Sizes = addList(Sizes, dS, 0)
            dW = []
            dC = []
            dS = []
            RBFNN = lambda X: RBFNet(X, Centers, Sizes, Weights)
        Error = Error + Err
    def RBF(X):
        R, Y = RBFNN(X)
        return Y
    return Error, RBF