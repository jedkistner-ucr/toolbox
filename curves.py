import numpy as np
# import matplotlib.pyplot as pl

def twobody(x, a, b):
    i = b * np.log(1+x*a)
    return i

def twobody1(x, a, b):
    i = (b/a) * np.log(1+x*a)
    return i

def line(x, m, b):
    i = m*x + b
    return i

# pl.figure()
# x = np.linspace(0, 5, 100)
# clr = ['r', 'b', 'g', 'y', 'k']
# for i in range(5):
#     y = twobody1(x, 1, 1+(1*i))
#     pl.plot(x, y, c = clr[i])
# for i in range(5):
#     y = twobody1(x,1+(1*i), 3)
#     pl.plot(x, y, c = clr[i], linestyle = ":")
# pl.show()

