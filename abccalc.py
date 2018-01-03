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

# Transform between bounded parameter space and continuous space
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
ind = [i for i in range(len(y)) if y[i]>1000]
lessx = [x[i] for i in range(len(y)) if i%10==0 or i in ind]
lessy = [y[i] for i in range(len(y)) if i%10==0 or i in ind]
sm_y = lowess(endog=y, exog=x, frac=6.0/len(x), return_sorted=False)
sm_y = np.array([max(1, _) for _ in sm_y])
# sm_bg = lowess(endog=yNIST-(Calc.Calculate())[1].data, exog=x, frac=6.0/len(x), return_sorted=False)


m = Calc.getCovMatrix()
keepm = (m[:,useInd])[useInd, :]
V = np.diag(keepm)

true_params = np.array([300,1,0.41385,0.7,-0.25,0.03,.001,130])
# Set up the lower and upper bounds on the parameters
# Set up desired window around initial parameters
V1 = [200,0.1,.005,0.2,.01,.01,.01,0.1,20]
lower = init_params-V1
upper = init_params+V1
#lower = np.array([0.0, 0.0, 1.535, 0.0, 200.0, -400, 125, -0.1, 1000])
#upper = np.array([1200.0, 1.5, 1.545, 0.5, 300.0, -250, 225, 0.1, 2000])
#lower = np.array([0,0,1.535,0,240,-340,170,-0.1,1000])
#upper = np.array([1200,1.5,1.545,0.5,250,-330,180,0.1,2000])

## Draw 
N = 500
drawb = 1
draws = np.exp(np.random.normal(loc=0,scale=1,size=N))
z = np.random.normal(size=(N,len(init_params)))
#can_params = z2par(z=z,lower=lower,upper=upper)
h = np.zeros((N,len(init_params)))
for i in range(N):
  h[i] = z2par(z=z[i],lower=lower,upper=upper)
can_params = h
# set up empty array for keepers
keep = np.zeros((0,init_params.shape[0]))
errs = np.zeros([0])
acc = 0
tot = 0
omega = np.diag(np.ones(20)*10)

# Calculate a B-spline basis for the range of scattering angles
unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(20-2)))
knots = augknt(unique_knots, 3)
objB = Bspline(knots, order=3)
B = objB.collmat(x)
del unique_knots, knots, objB

bob = np.matmul(np.matmul(B,omega),np.transpose(B))
print bob.shape

# Regress the observed data on the basis functions and obtain the smoothed residuals
model = sm.OLS(y,B).fit()
resids = y-model.predict(B)
sm_obs_r = lowess(endog=resids, exog=x, frac=6.0/len(x), return_sorted=False)
sm_obs_r = np.array([max(1, _) for _ in sm_obs_r])

thresh = 300000
for s in range(N):
  Calc.UpdateParameters(dict(zip(paramList, can_params[s,:]))) # Update calculator
  curr_mn = Calc.Calculate()[1].data
  curr_y = np.random.multivariate_normal(mean=curr_mn, cov = (bob + (draws[s])*np.diag(1+sm_y)))
  sm_curr_y = lowess(endog=curr_y, exog=x, frac=6.0/len(x), return_sorted=False)
  sm_curr_y = np.array([max(1, _) for _ in sm_curr_y])
  # Regress the current data draw on the basis functions
  m2 = sm.OLS(curr_y, B).fit()
  residCurr = curr_y - m2.predict(B)
  # Smooth the current residuals
  sm_curr_r = lowess(endog=residCurr, exog=x, frac=6.0/len(x), return_sorted=False)
  # Calculate weighted sum of squares
  ####### Compare smoothed current y to smoothed observed residuals
  #wss = np.sum(((sm_curr_r-sm_obs_r)**2)/sm_y)
  wss = np.sum(((sm_curr_y-sm_obs_r)**2)/sm_y)
  tot = tot + 1
  print tot
  if wss < thresh:
    keepParams = {key:Calc._parmDict[key] for key in paramList}
    keepi = np.array([[keepParams[key] for key in paramList]]) # Get Values
    keep = np.append(keep,keepi,axis=0)
    errs = np.append(errs,np.array([[wss]]))
    acc = acc+1
    #plt.plot(x,sm_obs_r,'k',x,curr_y,'r')
    #plt.show()

np.savetxt("keep7.txt",keep)
np.savetxt("errs7.txt",errs)    
print 'acceptance rate', acc/tot



'''

for (s in 1:N) {
  curr_mn <- Calc(params=can.params[, s], tth=X)
  e <- rnorm(length(Y),0,sqrt((1+drawb[s]*Y.sm)/drawt[s]))
  Y.sim <- curr_mn + e
  Y.sim.sm <- loess(Y.sim ~ X, span=sp)$fitted
  Y.sim.sm[Y.sim.sm < 0] <- 0
  err <- mean((Y.sm-Y.sim.sm)^2)
  if (err<500){
    keep<-cbind(keep,can.params[,s])
    errs<-c(errs, err)
    acc <- acc+1
  }
  if (s%%1000==0){
    print(s)
  }
  
}
'''


'''
# Set up the lower and upper bounds on the parameters
lower = np.array([200, 0, 0.41, 0.1, -0.35, 0.01, -0.1, 100])
upper = np.array([300, 2, 0.42, 1.5, -0.04, 0.1, 0.1, 200])

# Calculate a B-spline basis for the range of x
L = 19
unique_knots = np.percentile(a=x, q=np.linspace(0, 100, num=(L-1)))
knots = augknt(unique_knots, 3)
objB = Bspline(knots, order=3)
B = objB.collmat(x)
del unique_knots, knots, objB

# Generate the background process
gamma = np.array([370, 82, 105, 115, 44, 103, 75, 71, 117, 95, 104, 81, 119, 85, 88, 83, 77, 76, 48, 62])
true_BG = np.matmul(B, gamma)
# plt.plot(x, true_BG, 'k')
# plt.show()

# Set the true parameter values
Mstrain = 300   # Microstrain (0:0:Mustrain)
Size = 1        # Size (0:0:Size)
Io = 130        # Scale (:0:Scale)
l = 0.41385     # Lambda (:0:Lam)
tth0 = 0.001    # 2theta offset (:0:Zero)
U = 0.7         # Caglioti parameters
V = -0.25
W = 0.03
sigma_sq = 0.1
b = 0.7
true_params = np.array([Mstrain, Size, l, U, V, W, tth0, Io])

# Generate data based on the true parameter values
Calc.UpdateParameters(dict(zip(paramList, true_params))) # Update calculator
true_mn = true_BG + Calc.Calculate()[1].data
errors = sigma_sq*(1+b*true_mn)
y = np.random.normal(loc=true_mn, scale=np.sqrt(errors))
plt.plot(x, yNIST, 'k', x, y, 'r')
plt.show()

plt.plot(yNIST, y, 'wo')
plt.show()

# Save the data
csv = open("simulatedSI.csv", 'w')
for i in range(len(x)):
    row = str(x[i]) + ',' + str(y[i]) + '\n'
    csv.write(row)
csv.close()
'''
