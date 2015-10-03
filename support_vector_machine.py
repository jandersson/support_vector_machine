from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math

# Mister Global
POWER = 2
RADIAL_OMEGA = 1
SIGMOID_DELTA = 1
SIGMOID_K = 0.5

#Class from Jokull (thanks bro)
class kernels:
   def linear(x, y):
       xTrans = numpy.array(x).transpose()
       return xTrans.dot(y) + 1

   def polynomial(x, y):
       return math.pow(kernels.linear(x,y), POWER)

   def radial(x, y):
       xTrans = numpy.array(x).transpose()
       xy  = numpy.subtract(xTrans, y)
       normXY = numpy.linalg.norm(xy)
       return math.exp( -math.pow(normXY,2)/(2*math.pow(RADIAL_OMEGA,2)))

   def sigmoid(x, y):
       xTrans = numpy.array(x).transpose()
       return numpy.tanh(SIGMOID_K*xTrans.dot(y) - SIGMOID_DELTA)

def build_p_matrix(samples, kernel_fun):
    matrix = []
    x_i = []
    x_j = []
    for i in range(len(samples)):
        row = []
        for j in range(len(samples)):
            P_element = samples[i][2] * samples[j][2] * kernel_fun([samples[i][0], samples[i][1]],
                                                                   [samples[j][0], samples[j][1]])
            row.append(P_element)
        matrix.append(row)
    return matrix

classA = [(random.normalvariate(-1.5, 1),
          random.normalvariate(0.5, 1),
          1.0)
          for i in range(5)] + \
        [(random.normalvariate(1.5, 1),
         random.normalvariate(0.5, 1),
         1.0)
        for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5),
          random.normalvariate(-0.5, 0.5),
          -1.0)
         for i in range(10)]

data = classA + classB
random.shuffle(data)
#Jokull
data = [(-0.4817775866423481, -0.676196126111814, -1.0), (-1.3746653598765421, -1.2335653503955357, 1.0), (-0.04952991285713326, -0.477259823574604, -1.0), (-2.435302270702021, 0.5791628345129953, 1.0), (-0.23207345925812842, -0.829241385869637, -1.0), (-1.746084907233614, 0.9006137892797217, 1.0), (-0.3698769551877943, 1.9746639924223914, 1.0), (1.3807418306962262, 0.8160567394627515, 1.0), (0.6481506000006643, -0.2993683817580585, -1.0), (0.006408247677090779, -0.711255503088516, -1.0), (-0.3556075093525846, 0.6450331217049499, -1.0), (0.9314091400315766, -0.7871051525340638, -1.0), (0.86816793401546, -0.5298525064359068, 1.0), (-0.04468918689694391, -0.5181337266696356, -1.0), (1.2636537068237677, 1.1427041812767662, 1.0), (0.09043582719932093, 0.04380816810541566, -1.0), (0.012425436725393221, -0.6643420601684578, -1.0), (-2.0485964643706644, 0.6991611363135597, 1.0), (-2.8139016711910467, 0.17553854976255767, 1.0), (1.6300789364960653, 0.3280724548177084, 1.0)]
N = len(data)

pylab.hold(True)
pylab.plot([p[0] for p in classA],
          [p[1] for p in classA],
          'bo')
pylab.plot([p[0] for p in classB],
          [p[1] for p in classB],
          'ro')
pylab.show()

print(data)
print(data[1][2])

x_i = []
x_j = []
for sample in data:
    x_i.append(sample[0])
    x_j.append(sample[1])
print(x_i)

mP = build_p_matrix(data, kernels.polynomial)
mP = matrix(mP)
# print(fancy_mP)

# q vector is a N long vector with -1's
mQ = matrix([-1.0 for i in range(N)])
# print(mQ)
# h is a vector with all 0's
mH = matrix([0.0 for i in range(N)])
# print(mH)

# the G Matrix is an identity matrix with -1's
mG = (-1)*matrix(numpy.identity(N))
# print(mG)

# Make the call to qp
r = qp(mP, mQ, mG, mH)
a = list(r['x'])
a = [alpha if (alpha > 0.00001) else 0 for alpha in a]
print(a)

# print[(alpha < 0.000005) for alpha in a]
# Pick out the close-to-zero's in the alpha vector
# nonzero_alphas = [alpha for alpha in a if (alpha > 0.0005) else 0 for alpha in a]
nonzero_alphas = [alpha for alpha in a if (alpha > 0.0001)]
for index, sample in enumerate(data):
#     print(sample*a[index])
#     print(a[index])
    nonzero_datapoints = [sample for sample in data if (a[index] > 0)]
nonzero_datapoints = []
for index, sample in enumerate(data):
#     print(index)
    if a[index] > 0:
        nonzero_datapoints.append(sample)

#     nonzero_datapoints = [sample if (sample*a[index] > 0) else 0]
print(nonzero_datapoints)
print(nonzero_alphas)
# print(a)
# Save the corresponding X_i's (which has a nonzero alpha)
# x_i = [sample if for sample in data]
#generate new datapoints without a class and feed it into the indicator function which already knows about the non
#zero alphas and corresponding datapoints
new_n = len(nonzero_alphas)
xrange = numpy.arange(-4, 4, 0.05)
yrange = numpy.arange(-4, 4, 0.05)
# for index in range(new_n):
#     new_data = [(xrange[index], yrange[index])]
new_data = []
for i in range(new_n):
    new_data.append((xrange[i], yrange[i]))
# new_data = zip(xrange,yrange)
print(new_data)
# print(xrange)
def indicator(new_data, alphas, support_points, kernel_method):
    for i in range(len(alphas)):
        alphas[i]*support_points[i][2]*kernel_method([support_points[i][0], support_points[i][1]],
                                                     [new_data[i][0], new_data[i][1]])

# HERE BE ERRORS
# grid = matrix([
#         [indicator(new_data[:new_n], nonzero_alphas, nonzero_datapoints, kernels.polynomial) for y in yrange]
#         for x in xrange])
