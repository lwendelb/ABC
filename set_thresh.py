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

np.random.seed(112217)  # Set the seed for replicability

# Identify GPX file location
gpxFile = 'NIST_Si_HB2A_Ge115.gpx'

# Initialize the calculator
Calc = gsas.Calculator(GPXfile=gpxFile)
'''
paramList = [key for key in Calc._varyList if (key!=':0:DisplaceX' and key!='0::AUiso:0')]
gpxParams = {key:Calc._parmDict[key] for key in paramList}
init_params = np.array([gpxParams[key] for key in paramList])  # Get values
useInd = [np.asscalar(np.where(np.array(Calc._varyList)==par)[0]) for par in paramList]
'''
# Pull in the data
all_y = np.genfromtxt("manyy3.txt")
## Read the data from the calculator
x = Calc._tth
#y = Calc._Histograms[Calc._Histograms.keys()[0]]['Data'][1][:-1]
y = all_y[:,0]
high = [i for i in range(len(y)) if x[i]<140]
lessx = [x[i] for i in range(len(y)) if i in high]
lessy = [y[i] for i in range(len(y)) if i in high]
#ind = [i for i in range(len(y)) if y[i]>1000]
#lessx = [x[i] for i in range(len(y)) if i%10==0 or i in ind]
#lessy = [y[i] for i in range(len(y)) if i%10==0 or i in ind]
sm_y = lowess(endog=lessy, exog=lessx, frac=6.0/len(lessx), return_sorted=False)
sm_y = np.array([max(1, _) for _ in sm_y])
# sm_bg = lowess(endog=yNIST-(Calc.Calculate())[1].data, exog=x, frac=6.0/len(x), return_sorted=False)

# Calculate a B-spline basis for the range of scattering angles
unique_knots = np.percentile(a=lessx, q=np.linspace(0, 100, num=(20-2)))
knots = augknt(unique_knots, 3)
objB = Bspline(knots, order=3)
B = objB.collmat(lessx)
del unique_knots, knots, objB

# Regress the observed data on the basis functions and obtain the smoothed residuals
model = sm.OLS(lessy,B).fit()
resids = lessy-model.predict(B)
sm_obs_r = lowess(endog=resids, exog=lessx, frac=6.0/len(lessx), return_sorted=False)

# Calculate WSS for simulated "true" data
wss = np.zeros(1000)
for j in range(1000):
  i = j+1
  curr_y_full = all_y[:, i]
  curr_y = [curr_y_full[i] for i in range(len(curr_y_full)) if i in high]
  #sm_curr_y = lowess(endog=curr_y, exog=lessx, frac=6.0/len(lessx), return_sorted=False)
  #sm_curr_y = np.array([max(1, _) for _ in sm_curr_y])
  #less_all = [curr_y[_] for _ in range(len(y)) if _%10==0 or _ in ind]
  # Regress the current data draw on the basis functions
  #print curr_y.shape
  #print B.shape
  m2 = sm.OLS(curr_y, B).fit()
  residCurr = curr_y - m2.predict(B)
  # Smooth the current residuals
  sm_curr_r = lowess(endog=residCurr, exog=lessx, frac=6.0/len(lessx), return_sorted=False)
  # plt.plot(lessx,less_all)
  #plt.plot(lessx,sm_curr_r,'k', lessx,sm_obs_r, 'r')
  #plt.show()
  #wss[i] = np.sum(((sm_curr_r-sm_obs_r)**2)/sm_y)
  wss[j] = np.sum(((sm_curr_r-sm_obs_r)**2)/sm_y)
  print wss[j]
  
print np.mean(wss)
print np.min(wss),np.max(wss)
print np.percentile(a=wss,q=95)


#### with tail
'''
mean = 1314.10431725
min,max = 1080.38851512 1846.91907706
95 percentile = 1487.88587176
'''
### without tail
'''
1285.45473625
988.64638273 1915.29080294
1481.07356605
'''
