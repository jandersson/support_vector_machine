from cvxopt.solvers import qp
from cvxopt.base import matrix
from array import array

import numpy, pylab, random, math

## generating input data
classA = [(random.normalvariate(1, 1),
            random.normalvariate(0.5, 1),
            1.0)
           for i in range(5)] + \
         [(random.normalvariate(1.5, 1),
            random.normalvariate(0.5, 1),
            1.0)
           for i in range(5)];

classB = [(random.normalvariate(-1, 1),
            random.normalvariate(-0.5, 1),
            -1.0)
           for i in range(10)];

#Hard coded data
classA = [(0.9965606631942062, -0.5554064756635275, 1.0), (-0.4257474227660285, 0.12027119134640568, 1.0), (1.0685579193865484, 1.7257296825964354, 1.0), (1.045323475199482, -0.8711231251800222, 1.0), (0.549159919939763, 1.5881500878276624, 1.0), (2.0442145407307306, 1.6140637934176618, 1.0), (0.7890087908317421, 0.7036643434254063, 1.0), (0.37179721751008055, 2.1200080907899768, 1.0), (1.3818768555039853, 0.6776276672167125, 1.0), (1.3929594381459929, -0.48048225627197305, 1.0)]
classB = [(-2.2995482941558283, -1.0700711043993594, -1.0), (-0.2030449660422663, 1.0322082551171978, -1.0), (-2.8795010107069663, 0.05168709419608353, -1.0), (-2.414540711588119, -1.6575303869890203, -1.0), (0.6908965254702872, -0.9827822790074381, -1.0), (-0.5307462629348426, -0.35114912081275895, -1.0), (-0.8721199516085301, -1.580983996797356, -1.0), (-1.743754539812206, 0.3870688986767209, -1.0), (-0.9557368849879198, -1.7059133825957709, -1.0), (0.5517184529744592, -0.8129233391226747, -1.0)]

gen_data = classA + classB;
random.shuffle(gen_data);

pylab.figure();
pylab.hold(True);
pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo');
pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro');

## kernel function
def kernel_linear(x, y):
    "Linear kernel"
    temp = numpy.transpose(x);
    temp = numpy.dot(temp,y) + 1;
    return temp;

def kernel_polynomial(x, y):
    "Polynomial kernel"
    return math.pow(kernel_linear(x,y),POLY_POWER);

def kernel_radial(x, y):
    "Radial kernel"
    xTrans = numpy.array(x).transpose();
    xy  = numpy.subtract(xTrans, y);
    normXY = numpy.linalg.norm(xy);
    return math.exp(-math.pow(normXY,2)/(2*math.pow(RADIAL_SIGMA,2)));

def kernel_sigmoid(x, y):
    "Sigmoid kernel"
    k = .1;
    d = 1;
    xT = numpy.transpose(x);
    return math.tanh(k*numpy.dot(xT,y) - d);

KERNEL_FUNCTION = kernel_radial
RADIAL_SIGMA = 1
POLY_POWER = 2

temp = numpy.zeros(shape=[len(gen_data),len(gen_data)]);
##temp[0,0] = (1,2); ## probeer hier pair of tuple van te maken... ## zou nu gefixed moeten zijn

def buildP(data):
    for i in range(0,len(data)):
        for j in range(0,len(data)):
            temp[i,j] = KERNEL_FUNCTION(data[i][0:2],data[j][0:2]);
            temp[i,j] *= data[i][2];
            ## print "predict = ";
            ## print predict(data[i]);
            temp[i,j] *= data[j][2];
    return temp;

P = buildP(gen_data);

q = numpy.ones(len(gen_data))* -1.0;

G = numpy.identity(len(gen_data)) * -1;

h = numpy.zeros(len(gen_data));

## this should work by now as well
r  =  qp(matrix(P), matrix(q), matrix(G) , matrix(h))
alpha = list(r['x'])

## filter out extremely low values and set them to zero
counter = 0;
non_zero_alpha = [];

slack = False   ;
c = .5;

for i in range(0, len(alpha)):
    if alpha[i] < pow(10,-5):
        alpha[i] = 0;
    else:
        if (slack == False or (slack and alpha[i] < c)):
            non_zero_alpha.append([]);
            non_zero_alpha[counter].append(alpha[i]);
            non_zero_alpha[counter].append(gen_data[i][0]);
            non_zero_alpha[counter].append(gen_data[i][1]);
            non_zero_alpha[counter].append(gen_data[i][2]);
            counter = counter + 1;
##print non_zero_alpha;
##print "alpha"
##print non_zero_alpha[0][1:3];

def indicator(x,y):
    temp = 0;
    for i in range(0, len(non_zero_alpha)):
        k = KERNEL_FUNCTION([x,y],non_zero_alpha[i][1:3]);
        temp = temp + k * non_zero_alpha[i][3] * non_zero_alpha[i][0];
    return temp;


def draw_graph():
    ## draw decision boundary
    xrange = numpy.arange(-4, 4, 0.05)
    yrange = numpy.arange(-4, 4, 0.05)

    grid = matrix([[indicator(x,y)
                    for y in yrange]
                   for x in xrange])
    pylab.hold(True);
    pylab.plot([p[0] for p in classA],
               [p[1] for p in classA],
               'bo');
    pylab.plot([p[0] for p in classB],
               [p[1] for p in classB],
               'ro');

    pylab.contour(xrange, yrange, grid,
                  (-1.0, 0.0, 1.0),
                  colors = ('red', 'black', 'blue'),
                  linewidths = (1, 3, 1));
    pylab.show();

def make_svm():
    global KERNEL_FUNCTION
    funcs = [kernel_linear, kernel_polynomial, kernel_radial]

    for kernel in funcs:
        KERNEL_FUNCTION = kernel
        draw_graph()

make_svm()