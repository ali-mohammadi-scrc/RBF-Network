from RBFNetwork import RBFNetwork
from RBFNetwork import gnuplotOut
import random as rand
rand.seed(3.1415)

print('Training model for f(x) = 2 * x^2 - 0.7 * x - 1 for x in [-5, 5]')
f = lambda x: [2 * x[0] ** 2 - 0.7 * x[0] - 1]
NTrainingPatterns = 200
NTestPatterns = 100
X = [[rand.random() * 10 - 5] for x in range(NTrainingPatterns)]
TrainingPatterns = [(x, f(x)) for x in X]
X = [[rand.random() * 10 - 5] for x in range(NTestPatterns)]
TestPatterns = [(x, f(x)) for x in X]
N = 1;
K = 10;
M = 1;
Weights = []
LearningRateC = 0.02
LearningRateS = 0.01
LearningRateW = 0.001
RandomSeed = 7
MaxSteps = 1500
Batch = 0
##
E, F = RBFNetwork (N, M, K, Weights, TrainingPatterns, LearningRateC, LearningRateS, LearningRateW, RandomSeed, MaxSteps, Batch)
if type(TestPatterns) == str:
    TestPatterns = readPatterns(TestPatterns, Layers[0], Layers[-1])
TestErr = [sum([(a - b) ** 2 for a, b in zip(p[1], F(p[0]))]) for p in TestPatterns]
print('Total error for test patterns: ' + str(sum(TestErr)))

gnuplotOut('Learning_Curve.plt', list(range(len(E))), E, 'No. Patterns( *' + str(len(TrainingPatterns)) + ')', 'Error', 'Learning Curve', 'line')
list.sort(TrainingPatterns)
list.sort(TestPatterns)
gnuplotOut('Train.plt', [p[0][0] for p in TrainingPatterns], [p[1][0] for p in TrainingPatterns], 'X', 'Y', 'Training Patterns', 'points')
gnuplotOut('Test.plt', [p[0][0] for p in TestPatterns], [F(p[0])[0] for p in TestPatterns], 'X', 'Y', 'Model Performance on Test Patterns', 'points')
