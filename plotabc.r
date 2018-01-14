data = read.table('keep16.txt')
errors = read.table('errs16.txt')
t=1482
hist(data[errors<t,1])
hist(data[errors<t,2])
hist(data[errors<t,3])
hist(data[errors<t,4])
hist(data[errors<t,5])
hist(data[errors<t,6])
hist(data[errors<t,7])
hist(data[errors<t,8])
hist(data[errors<t,9])
plot(density(data[errors<t,1]))
plot(density(data[errors<t,2]))
plot(density(data[errors<t,3]))
plot(density(data[errors<t,4]))
plot(density(data[errors<t,5]))
plot(density(data[errors<t,6]))
plot(density(data[errors<t,7]))
plot(density(data[errors<t,8]))
plot(density(data[errors<t,9]))

colMeans(data[errors<t,])

'''
paramList = ['0:0:Mustrain;i', '0:0:Size;i', ':0:Lam', ':0:SH/L', ':0:U', ':0:V', ':0:W', ':0:Zero', ':0:Scale']

Mstrain = 505   # Microstrain (0:0:Mustrain)
Size = 0.45     # Size (0:0:Size)
l = 1.54        # Lambda (:0:Lam)
SH = 0.1        # Axial divergence (:0:SH/L)
U = 248         # Caglioti parameters
V = -333
W = 177
tth0 = 0.065    # 2theta offset (:0:Zero)
Io = 1756       # Scale (:0:Scale)
sigma_sq = 0.1
b = 1.0
'''
