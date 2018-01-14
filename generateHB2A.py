from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys
sys.path.insert(1, 'C:\Users\Laura\Documents\ABC')
sys.path.insert(3, r'C:\Users\Laura\g2conda\GSASII\bindist')
sys.path.insert(2, r'C:\Users\Laura\g2conda\GSASII')

# Import entire modules
import numpy as np
import matplotlib.pyplot as plt

# Source functions
import GSAS_Calculator_Opt as gsas         # GSAS calculator
from bspline import Bspline                # Bspline function
from splinelab import augknt               # Bspline helper function
from timeit import default_timer as timer  # Timing function
from scipy.stats import norm               # Normal distribution
import statsmodels.api as sm               # Library for lowess smoother
lowess = sm.nonparametric.lowess           # Lowess smoothing function

np.random.seed(121117)  # Set the seed for replicability

# Identify GPX file location and parameter list
gpxFile = 'NIST_Si_HB2A_Ge115.gpx'
paramList = ['0:0:Mustrain;i', '0:0:Size;i', ':0:Lam', ':0:SH/L', ':0:U', ':0:V', ':0:W', ':0:Zero', ':0:Scale']

# Initialize the calculator
Calc = gsas.Calculator(GPXfile=gpxFile)
# paramList = [key for key in Calc._varyList if (key!=':0:DisplaceX' and key!='0::AUiso:0')]
gpxParams = {key:Calc._parmDict[key] for key in paramList}

# Initialize the calculator and pull out range of scattering angles
Calc = gsas.Calculator(GPXfile=gpxFile)
x = Calc._tth
yOrig = Calc._Histograms[Calc._Histograms.keys()[0]]['Data'][1][:-1]

# Calculate a B-spline basis for the range of x
L = 19
unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(L-1)))
knots = augknt(unique_knots, 3)
objB = Bspline(knots, order=3)
B = objB.collmat(x)
del unique_knots, knots, objB

# # Get ballpark background estimates
# BG = yOrig-(Calc.Calculate())[1].data
# m1 = sm.OLS(BG[0:cut], B).fit()
# print m1.params

# Generate the background process
gamma = np.array([46, 35, 39, 33, 32, 32, 33, 30, 32, 30, 31, 30, 30, 30, 31, 30, 33, 30, 66, 50])
true_BG = np.matmul(B, gamma)
# plt.plot(x, true_BG, 'k')
# plt.show()

# Set the true parameter values
Mstrain = 505   # Microstrain (0:0:Mustrain)
Size = 0.45     # Size (0:0:Size)
SH = 0.1        # Axial divergence (:0:SH/L)
Io = 1756       # Scale (:0:Scale)
l = 1.54        # Lambda (:0:Lam)
tth0 = 0.065    # 2theta offset (:0:Zero)
U = 248         # Caglioti parameters
V = -333
W = 177
sigma_sq = 0.6
b = 1.0
true_params = np.array([Mstrain, Size, l, SH, U, V, W, tth0, Io])

# Generate data based on the true parameter values
Calc.UpdateParameters(dict(zip(paramList, true_params))) # Update calculator
#true_mn = true_BG + Calc.Calculate()[1].data
true_mn = Calc.Calculate()[1].data
errors = sigma_sq*(1+b*true_mn)
y = np.random.normal(loc=true_mn, scale=np.sqrt(errors))
plt.plot(x, yOrig-true_BG, 'k', x, y, 'r')
plt.show()

plt.plot(yOrig-true_BG, y, 'wo')
plt.show()

y = np.zeros((len(yOrig),1001))
for i in range(1001):
  y[:,i] = np.random.normal(loc=true_mn, scale=np.sqrt(errors))
np.savetxt("manyy3.txt",y)

'''
# Save the data
csv = open("simulatedHB2A.csv", 'w')
for i in range(len(x)):
    row = str(x[i]) + ',' + str(y[i]) + '\n'
    csv.write(row)
csv.close()
'''
