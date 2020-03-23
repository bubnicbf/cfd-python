import numpy as np

def GaussLegendre(num_points):
    """
    Return points and weights for Gauss-Legendre rules on [-1,1].
    The no of points implemented are 1-20, 32, 64, 96, 100, 128, 256,
    512, 1024.
    """
    n = num_points
    points  = np.zeros(n)
    weights = np.zeros(n)

    if n > 1:
        try:
            x[n]  # x is defined below (global variable)
        except KeyError:
            raise ValueError(
                'Gauss-Legendre rule with %d points not available' % n)

    if n == 1:
        points[0] = 0
        weights[0] = 2
    elif n % 2 == 0:
        for i in range(len(x[n])):
            points[n//2+i] = x[n][i]
            points[n//2-1-i] = -x[n][i]
            weights[n//2+i] = w[n][i]
            weights[n//2-1-i] = w[n][i]
    else:
        for i in range(len(x[n])):
            points[n//2+i] = x[n][i]
            points[n//2-i] = -x[n][i]
            weights[n//2+i] = w[n][i]
            weights[n//2-i] = w[n][i]
    return points, weights

x = {}
w = {}
x[2] = [0.5773502691896257645091488]
w[2] = [1.0000000000000000000000000]

x[4] = [0.3399810435848562648026658,0.8611363115940525752239465]
w[4] = [0.6521451548625461426269361,0.3478548451374538573730639]

x[6] = [0.2386191860831969086305017,0.6612093864662645136613996,0.9324695142031520278123016]
w[6] = [0.4679139345726910473898703,0.3607615730481386075698335,0.1713244923791703450402961]

x[8] = [0.1834346424956498049394761,0.5255324099163289858177390,0.7966664774136267395915539,0.9602898564975362316835609]
w[8] = [0.3626837833783619829651504,0.3137066458778872873379622,0.2223810344533744705443560,0.1012285362903762591525314]

x[10] = [0.1488743389816312108848260,0.4333953941292471907992659,0.6794095682990244062343274,0.8650633666889845107320967,0.9739065285171717200779640]
w[10] = [0.2955242247147528701738930,0.2692667193099963550912269,0.2190863625159820439955349,0.1494513491505805931457763,0.0666713443086881375935688]

x[12] = [0.1252334085114689154724414,0.3678314989981801937526915,0.5873179542866174472967024,0.7699026741943046870368938,0.9041172563704748566784659,0.9815606342467192506905491]
w[12] = [0.2491470458134027850005624,0.2334925365383548087608499,0.2031674267230659217490645,0.1600783285433462263346525,0.1069393259953184309602547,0.0471753363865118271946160]

x[3] = [0.0000000000000000000000000,0.7745966692414833770358531]
w[3] = [0.8888888888888888888888889,0.5555555555555555555555556]

x[5] = [0.0000000000000000000000000,0.5384693101056830910363144,0.9061798459386639927976269]
w[5] = [0.5688888888888888888888889,0.4786286704993664680412915,0.2369268850561890875142640]

x[7] = [0.0000000000000000000000000,0.4058451513773971669066064,0.7415311855993944398638648,0.9491079123427585245261897]
w[7] = [0.4179591836734693877551020,0.3818300505051189449503698,0.2797053914892766679014678,0.1294849661688696932706114]

x[9] = [0.0000000000000000000000000,0.3242534234038089290385380,0.6133714327005903973087020,0.8360311073266357942994298,0.9681602395076260898355762]
w[9] = [0.3302393550012597631645251,0.3123470770400028400686304,0.2606106964029354623187429,0.1806481606948574040584720,0.0812743883615744119718922]