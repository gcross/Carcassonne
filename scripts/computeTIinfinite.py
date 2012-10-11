from __future__ import division, print_function
from math import pi
from scipy.special import ellipeinc
import sys

J = float(sys.argv[1])

Gamma = 1
lam = J/(2*Gamma)
theta = 4*lam/(1+lam)**2
print(ellipeinc(pi/2,theta)/(pi/2),1+lam)
print("{:.15f}".format(-Gamma*2/pi*(1+lam)*ellipeinc(pi/2,theta)))
