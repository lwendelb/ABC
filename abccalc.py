from __future__ import division  # Use "real" division

# Add another filepath to the Python path
import sys

sys.path.insert(1, 'C:\Users\Laura\Documents\ABC')
sys.path.insert(3, r'C:\Users\Laura\g2conda\GSASII\bindist')
sys.path.insert(2, r'C:\Users\Laura\g2conda\GSASII')


# Import entire modules
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer  # Timing function
from scipy.stats import truncnorm

# Source functions
import GSAS_Calculator_Opt as gsas         # GSAS calculator
from bspline import Bspline                # Bspline function
from splinelab import augknt               # Bspline helper function
from timeit import default_timer as timer  # Timing function
from scipy.stats import norm               # Normal distribution
import statsmodels.api as sm               # Library for lowess smoother
lowess = sm.nonparametric.lowess           # Lowess smoothing function

#np.random.seed(112217)  # Set the seed for replicability
np.random.seed(112217+2000)

# Transform between bounded parameter space and continuous space
'''
def z2par(z, lower, upper, grad=False):
    if (grad):
        d = (upper-lower)*norm.pdf(z)
        return d
    else:
        par = lower + (upper-lower)*norm.cdf(z)
        # Fudge the parameter value if we've hit either boundary
        par[np.array([par[j]==upper[j] for j in range(len(par))])] -= 1e-10
        par[np.array([par[j]==lower[j] for j in range(len(par))])] += 1e-10
        #par[np.array([par[j]==upper[j] for j in range(par.shape[1])])] -= 1e-10
        #par[np.array([par[j]==lower[j] for j in range(par.shape[1])])] += 1e-10
        return par
'''

# Identify GPX file location
gpxFile = 'NIST_Si_HB2A_Ge115.gpx'

# Initialize the calculator
Calc = gsas.Calculator(GPXfile=gpxFile)
## Declare the parameters to be refined 
paramList = ['0:0:Mustrain;i', '0:0:Size;i', ':0:Lam', ':0:SH/L', ':0:U', ':0:V', ':0:W', ':0:Zero', ':0:Scale']
gpxParams = {key:Calc._parmDict[key] for key in paramList}
init_params = np.array([gpxParams[key] for key in paramList])  # Get values
useInd = [np.asscalar(np.where(np.array(Calc._varyList)==par)[0]) for par in paramList]

print 'init', init_params

# Pull in the data
## Read the data from the calculator
x = Calc._tth
y = Calc._Histograms[Calc._Histograms.keys()[0]]['Data'][1][:-1]
high = [i for i in range(len(y)) if x[i]<140]
lessx = [x[i] for i in range(len(y)) if i in high]
lessy = [y[i] for i in range(len(y)) if i in high]
x = lessx
y = lessy
sm_y = lowess(endog=y, exog=x, frac=6.0/len(x), return_sorted=False)
sm_y = np.array([max(1, _) for _ in sm_y])
# sm_bg = lowess(endog=yNIST-(Calc.Calculate())[1].data, exog=x, frac=6.0/len(x), return_sorted=False)


m = Calc.getCovMatrix()
keepm = (m[:,useInd])[useInd, :]
V = np.diag(keepm)
se = np.sqrt(V)
c = 2

# Set up the lower and upper bounds on the parameters
# Set up desired window around initial parameters
#### Tight Bounds
#V1 = [200,0.1,.005,0.09,5,5,5,0.08,20]
#V2 = [200,0.1,.005,0.09,5,5,5,0.08,20]
#### Loosen up Scale Param
#V1 = [200,0.1,.005,0.09,5,5,5,0.08,100]
#V2 = [200,0.1,.005,0.09,5,5,5,0.08,100]
## Susheela's wide bounds
lower_wide = np.array([0.0, 0.0, 1.53, 0.0, 200.0, -400, 125, -0.1, 1000])
upper_wide = np.array([1200.0, 1.5, 1.55, 0.5, 300.0, -250, 225, 0.1, 2000])
#lower = np.array([0,0,1.53,0,240,-340,170,-0.1,1000])
#upper = np.array([1200,1.5,1.55,0.5,250,-330,180,0.1,2000])
#lower = init_params-V1
#upper = init_params+V2
#### +- 1 Rietveld standard error
#lower = np.maximum(init_params-c*se,lower_wide)
#upper = np.minimum(init_params+c*se,upper_wide)
#lower = np.array([354.81434502,0,1.53300981,0.08905228,238.18375496,-349.57450027,169.59124779,0.06391135,1747.37206718])
#upper = np.array([655.47972590,1.04344387,1.54158028,0.09285496,257.24508190,-317.29688361,184.01575712,0.06995091,1763.87610124])
lower = lower_wide
upper = upper_wide


print lower
print upper

## Draw parameters
N = 2000
drawb = 1
draws = np.exp(np.random.normal(loc=0,scale=1,size=N))
# initialize array for candidate parameters
can_params = np.zeros((N,len(init_params)))
# Fill array with truncated normal draws
for i in range(len(init_params)):
  # normalize bounds
  a, b = (lower[i] - init_params[i]) / se[i], (upper[i] - init_params[i]) / se[i]
  # draw parameters
  can_params[:,i]=se[i]*truncnorm.rvs(a,b,size=N)+init_params[i]

# set up empty arrays for parameters and errors
keep = np.zeros((N,init_params.shape[0]))
errs = np.zeros(N)

omega = np.diag(np.ones(20)*10)

# Calculate a B-spline basis for the range of scattering angles
unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(20-2)))
knots = augknt(unique_knots, 3)
objB = Bspline(knots, order=3)
B = objB.collmat(x)
del unique_knots, knots, objB

bob = np.matmul(np.matmul(B,omega),np.transpose(B))

# Regress the observed data on the basis functions and obtain the smoothed residuals
model = sm.OLS(y,B).fit()
resids = y-model.predict(B)
sm_obs_r = lowess(endog=resids, exog=x, frac=6.0/len(x), return_sorted=False)

# Start Timer
tic = timer()

for s in range(N):
  #Update Calculator
  Calc.UpdateParameters(dict(zip(paramList, can_params[s,:])))
  curr_mn_long = Calc.Calculate()[1].data
  curr_mn = [curr_mn_long[i] for i in range(len(curr_mn_long)) if i in high]
  curr_y = np.random.multivariate_normal(mean=curr_mn, cov = (bob + (draws[s])*np.diag(1+sm_y)))
  
  # Regress the current data draw on the basis functions
  m2 = sm.OLS(curr_y, B).fit()
  residCurr = curr_y - m2.predict(B)
  
  # Smooth the current residuals
  sm_curr_r = lowess(endog=residCurr, exog=x, frac=6.0/len(x), return_sorted=False)
  
  # Calculate weighted sum of squares
  ####### Compare smoothed candidate residuals to smoothed observed residuals (wss)
  errs[s] = np.sum(((sm_curr_r-sm_obs_r)**2)/sm_y)
  
  keepParams = {key:Calc._parmDict[key] for key in paramList}
  keep[s,:] = np.array([[keepParams[key] for key in paramList]]) # Get Values
  
  if s % 50 is 0:
    print "Iteration %d, elapsed time %03.2f min" %(s, (timer()-tic)/60)
  #plt.plot(x,sm_obs_r,'k',x,sm_curr_r,'r')
  #plt.show()

toc = timer()
print "total elapsed time %03.2f min" %((toc-tic)/60)

np.savetxt("keep17.txt",keep)
np.savetxt("errs17.txt",errs)    

