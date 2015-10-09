from cvxopt.solvers import qp
from cvxopt.base import matrix
from array import array

import numpy, pylab, random, math

KERNEL_FUNCTION = None
RADIAL_SIGMA = 1
POLY_POWER = 2

def plot_datapoints(classA, classB):
    '''Plots the data points without drawing any boundaries'''
    pylab.figure()
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo')
    pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro')
    pylab.show()


## generating input data
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

#Hard coded data
 # classA = [(0.9965606631942062, -0.5554064756635275, 1.0), (-0.4257474227660285, 0.12027119134640568, 1.0), (1.0685579193865484, 1.7257296825964354, 1.0), (1.045323475199482, -0.8711231251800222, 1.0), (0.549159919939763, 1.5881500878276624, 1.0), (2.0442145407307306, 1.6140637934176618, 1.0), (0.7890087908317421, 0.7036643434254063, 1.0), (0.37179721751008055, 2.1200080907899768, 1.0), (1.3818768555039853, 0.6776276672167125, 1.0), (1.3929594381459929, -0.48048225627197305, 1.0)]
 # classB = [(-2.2995482941558283, -1.0700711043993594, -1.0), (-0.2030449660422663, 1.0322082551171978, -1.0), (-2.8795010107069663, 0.05168709419608353, -1.0), (-2.414540711588119, -1.6575303869890203, -1.0), (0.6908965254702872, -0.9827822790074381, -1.0), (-0.5307462629348426, -0.35114912081275895, -1.0), (-0.8721199516085301, -1.580983996797356, -1.0), (-1.743754539812206, 0.3870688986767209, -1.0), (-0.9557368849879198, -1.7059133825957709, -1.0), (0.5517184529744592, -0.8129233391226747, -1.0)]

gen_data = classA + classB
random.shuffle(gen_data)

print("classA = " + str(classA))
print("classB = " + str(classB))
# plot_datapoints(classA, classB)

pylab.figure()
pylab.hold(True)
pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo')
pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro')

## kernel function
def kernel_linear(x, y):
    "Linear kernel"
    temp = numpy.transpose(x)
    temp = numpy.dot(temp,y) + 1
    return temp;

def kernel_polynomial(x, y):
    "Polynomial kernel"
    return math.pow(kernel_linear(x,y),POLY_POWER)

def kernel_radial(x, y):
    "Radial kernel"
    xTrans = numpy.array(x).transpose()
    xy  = numpy.subtract(xTrans, y)
    normXY = numpy.linalg.norm(xy)
    return math.exp(-math.pow(normXY,2)/(2*math.pow(RADIAL_SIGMA,2)))

def kernel_sigmoid(x, y):
    "Sigmoid kernel"
    k = .1
    d = 1
    xT = numpy.transpose(x)
    return math.tanh(k*numpy.dot(xT,y) - d)



temp = numpy.zeros(shape=[len(gen_data),len(gen_data)])
##temp[0,0] = (1,2); ## probeer hier pair of tuple van te maken... ## zou nu gefixed moeten zijn

def buildP(data):
    for i in range(0,len(data)):
        for j in range(0,len(data)):
            temp[i,j] = KERNEL_FUNCTION(data[i][0:2],data[j][0:2])
            temp[i,j] *= data[i][2]
            temp[i,j] *= data[j][2]
    return temp

def indicator(x,y, alphas):
    non_zero_alpha = alphas
    temp = 0
    for i in range(0, len(non_zero_alpha)):
        k = KERNEL_FUNCTION([x,y],non_zero_alpha[i][1:3]);
        temp = temp + k * non_zero_alpha[i][3] * non_zero_alpha[i][0];
    return temp


def draw_graph(alphas, plot_title="I forgot to add a title"):
    ## draw decision boundary
    xrange = numpy.arange(-4, 4, 0.05)
    yrange = numpy.arange(-4, 4, 0.05)

    grid = matrix([[indicator(x,y, alphas)
                    for y in yrange]
                   for x in xrange])
    pylab.hold(True)
    pylab.plot([p[0] for p in classA],
               [p[1] for p in classA],
               'bo')
    pylab.plot([p[0] for p in classB],
               [p[1] for p in classB],
               'ro')

    pylab.contour(xrange, yrange, grid,
                  (-1.0, 0.0, 1.0),
                  colors = ('red', 'black', 'blue'),
                  linewidths = (1, 3, 1))

    pylab.title(plot_title)

    pylab.show()

def make_svm(kernel_func, title="I forgot the title"):
    global KERNEL_FUNCTION
    KERNEL_FUNCTION = kernel_func

    P = buildP(gen_data)

    q = numpy.ones(len(gen_data)) * -1.0

    G = numpy.identity(len(gen_data)) * -1

    h = numpy.zeros(len(gen_data))

    r  =  qp(matrix(P), matrix(q), matrix(G) , matrix(h))
    alpha = list(r['x'])

    counter = 0
    non_zero_alpha = []

    slack = False
    c = .5

    for i in range(0, len(alpha)):
        if alpha[i] < pow(10,-5):
            alpha[i] = 0
        else:
            if (slack == False or (slack and alpha[i] < c)):
                non_zero_alpha.append([])
                non_zero_alpha[counter].append(alpha[i])
                non_zero_alpha[counter].append(gen_data[i][0])
                non_zero_alpha[counter].append(gen_data[i][1])
                non_zero_alpha[counter].append(gen_data[i][2])
                counter = counter + 1

    draw_graph(non_zero_alpha, title)

#Dataset 1
# classA = [()]
# classA = [(-2.2828787943926274, 0.44835710394916284, 1.0), (-2.125398898010725, -0.45113480424004326, 1.0), (-1.5314008989189787, 1.121994625294799, 1.0), (-0.5527794901180547, 1.907534985208719, 1.0), (-1.396090933219027, -1.0412011160096786, 1.0), (3.6677469904725486, 2.9418683244810397, 1.0), (0.34332488643645553, 0.8942737778773645, 1.0), (0.11159791582497913, 1.2996866332961652, 1.0), (1.4253263730872077, 0.08841272114916993, 1.0), (1.7461087029329354, 1.0783424830514856, 1.0)]
# classB = [(0.804561163986563, -0.17560532507727872, -1.0), (0.33370960766449936, -1.2610925804772797, -1.0), (-1.032094989483795, -0.058026183949750176, -1.0), (0.5827368901024689, -0.5249499772206949, -1.0), (0.6203351769864921, -0.2959674835312266, -1.0), (0.30839856697042384, -1.349390495186868, -1.0), (0.011659437568314867, -0.6051170170786636, -1.0), (-0.5575258798519777, -0.6946926228344402, -1.0), (0.1107254394268112, -0.4229103105186625, -1.0), (0.28361044347328696, -0.8628854859682875, -1.0)]

#Dataset 2
# classA = [(-1.73125249588547, 0.9579578534394011, 1.0), (-1.5500452250898349, -1.4348196474246115, 1.0), (-1.857349650041414, 0.012208370657844081, 1.0), (-1.3790177380117168, -0.6191964622123798, 1.0), (-1.3331638494255542, 1.301264466094505, 1.0), (1.330360434701353, 1.542772391082639, 1.0), (1.956116259396972, -0.47553423654400173, 1.0), (2.218565718883157, -0.3915284750754332, 1.0), (2.1146299871174254, -0.5977783486801405, 1.0), (2.101648190232107, 0.6321571174694021, 1.0)]
# classB = [(-0.21231472129241716, 0.13371318207525829, -1.0), (-0.37141033175387345, -0.9379777679902823, -1.0), (0.3928175825155068, -1.3217298531236463, -1.0), (-0.8536334636562778, -0.13183627313510443, -1.0), (-0.2990777351608072, -0.03257917618305756, -1.0), (0.38129491833584883, -0.8503947907630562, -1.0), (0.06451781368143289, 0.1570007537193422, -1.0), (0.5790023025867382, -0.6927454565219274, -1.0), (-0.5639766863148532, -0.24658877912087496, -1.0), (0.5612689124849574, -0.3950638455686839, -1.0)]

gen_data = classA + classB
random.shuffle(gen_data)

make_svm(kernel_linear, "Linear Kernel")

# Draw some polynomial kernel SVMs
POLY_POWER = 2
make_svm(kernel_polynomial, "Polynomial Kernel (p = 2)")
POLY_POWER = 3
make_svm(kernel_polynomial, "Polynomial Kernel (p = 3)")
POLY_POWER = 5
make_svm(kernel_polynomial, "Polynomial Kernel (p = 5)")

#Draw some Radial kernel SVMs
RADIAL_SIGMA = 0.5 #sets sigma^2 = 0.25
make_svm(kernel_radial, "Radial/Gaussian Kernel (sigma^2 = 0.25)")
RADIAL_SIGMA = 0.7071 #sets sigma^2 = 0.50
make_svm(kernel_radial, "Radial/Gaussian Kernel (sigma^2 = 0.50)")
RADIAL_SIGMA = 1
make_svm(kernel_radial, "Radial/Gaussian Kernel (sigma^2 = 1)")
RADIAL_SIGMA = 2
make_svm(kernel_radial, "Radial/Gaussian Kernel (sigma^2 = 2)")

