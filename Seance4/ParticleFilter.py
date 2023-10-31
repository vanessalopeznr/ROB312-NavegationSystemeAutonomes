"""
TP particle filter for mobile robots localization

authors: Goran Frehse, David Filliat, Nicolas Merlinge
"""
import matplotlib.cm as cm
import matplotlib as mpl 
from math import sin, cos, atan2, pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
seed = 123456
np.random.seed(seed)

import os
try:
    os.makedirs("outputs")
except:
    pass

# ---- Simulator class (world, control and sensors) ----

class Simulation:
    def __init__(self, Tf, dt_pred, xTrue, QTrue, xOdom, Map, RTrue, dt_meas):
        self.Tf = Tf
        self.dt_pred = dt_pred
        self.nSteps = int(np.round(Tf/dt_pred))
        self.QTrue = QTrue
        self.xTrue = xTrue
        self.xOdom = xOdom
        self.Map = Map
        self.RTrue = RTrue
        self.dt_meas = dt_meas
        
    # return true control at step k
    def get_robot_control(self, k):
        # generate  sin trajectory
        u = np.array([[0, 0.025,  0.1*np.pi / 180 * sin(3*np.pi * k / self.nSteps)]]).T
        return u
    
    
    # simulate new true robot position
    def simulate_world(self, k):
        dt_pred = self.dt_pred
        u = self.get_robot_control(k)
        self.xTrue = tcomp(self.xTrue, u, dt_pred)
        self.xTrue[2, 0] = angle_wrap(self.xTrue[2, 0])
    
    
    # computes and returns noisy odometry
    def get_odometry(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*2 + k)
        
        # Model
        dt_pred = self.dt_pred
        u = self.get_robot_control(k)
        xnow = tcomp(self.xOdom, u, dt_pred)
        uNoise = np.sqrt(self.QTrue) @ np.random.randn(3)
        uNoise = np.array([uNoise]).T
        xnow = tcomp(xnow, uNoise, dt_pred)
        self.xOdom = xnow
        u = u + dt_pred*uNoise
        return xnow, u


    # generate a noisy observation of a random feature
    def get_observation(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*3 + k)

        # Model
        if k*self.dt_pred % self.dt_meas == 0:
            notValidCondition = False # False: measurement valid / True: measurement not valid
            #notValidCondition = False if k < 250 or k > 350 else True
            if notValidCondition:
                z = None
                iFeature = None
            else:
                iFeature = np.random.randint(0, self.Map.shape[1] - 1)
                zNoise = np.sqrt(self.RTrue) @ np.random.randn(2)
                zNoise = np.array([zNoise]).T
                z = observation_model(self.xTrue, iFeature, self.Map) + zNoise.T
                z = z.T
                z[1, 0] = angle_wrap(z[1, 0])
        else:
            z = None
            iFeature = None
        return [z, iFeature]



# ---- Particle Filter: model functions ----


# evolution model (f)
def motion_model(x, u, dt_pred):
    # x: estimated state (x, y, heading)
    # u: control input (Vx, Vy, angular rate)
    x_k=x.reshape(3,1)
    xPred=tcomp(x_k,u,dt_pred)

    return xPred


# observation model (h):  we use the state to estimate sensor measurements that represents a vector of predicted sensor measurements y
def observation_model(xVeh, iFeature, Map):
    # xVeh: vecule state
    # iFeature: observed amer index
    # Map: map of all amers
    
    xP=Map[:,iFeature]
    xVeh=np.squeeze(xVeh)
    z=np.array([sqrt((xP[0]-xVeh[0])**2+(xP[1]-xVeh[1])**2),atan2((xP[1]-xVeh[1]),(xP[0]-xVeh[0]))-xVeh[2]])
    
    return z


# ---- particle filter implementation ----

# Particle filter resampling
def re_sampling(px, pw):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / nParticles)
    re_sample_id = base + np.random.uniform(0, 1 / nParticles)
    indexes = []
    ind = 0
    for ip in range(nParticles):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
#    pw = pw[indexes]
    
    # Normalization
    pw = np.ones(pw.shape)
    pw = pw / np.sum(pw)

    return px, pw


    # Particle filter resampling
def re_sampling_sys(px, pw):
    """
    systematic resampling
    """
    j=0
    w_cum = pw[j]
    re_sample_id = np.random.rand() / nParticles
    indexes = []

    for ip in range(nParticles):
        while re_sample_id > w_cum:
            j += 1
            w_cum += pw[j]
        indexes.append(j)
        re_sample_id += 1 / nParticles

    px = px[:, indexes]
#    pw = pw[indexes]
    
    # Normalization
    pw = np.ones(pw.shape)
    pw = pw / np.sum(pw)

    return px, pw


# ---- Utils functions ----

# Init displays
show_animation = True
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
ax3 = plt.subplot(3, 2, 2)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 6)


# fit angle between -pi and pi
def angle_wrap(a):
    if (a > np.pi):
        a = a - 2 * pi
    elif (a < -np.pi):
        a = a + 2 * pi
    return a


# composes two transformations
def tcomp(tab, tbc, dt):
    assert tab.ndim == 2 # eg: robot state [x, y, heading]
    assert tbc.ndim == 2 # eg: robot control [Vx, Vy, angle rate]
    #dt : time-step (s)
    angle = tab[2, 0] + dt * tbc[2, 0] # angular integration by Euler

    angle = angle_wrap(angle)
    s = sin(tab[2, 0])
    c = cos(tab[2, 0])
    position = tab[0:2] + dt * np.array([[c, -s], [s, c]]) @ tbc[0:2] # position integration by Euler
    out = np.vstack((position, angle))

    return out


def plotParticles(simulation, k, iFeature, hxTrue, hxOdom, hxEst, hxError, hxSTD, save = True):
    # simulation : Simulation object (containing world simulation and sensors)
    # k : current time-step
    # iFeature : index of current emitting amer 
    # hxTrue : true trajectory
    # hxOdom : odometric trajectory
    # hxEst : estimated trajectory
    # hxError : error (basically "hxEst - hxTrue")
    # hxSTD : standard deviation on estimate
    # save : True to save a figure as an image
        
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

    ax1.cla()

    # Plot true landmark and trajectory
    ax1.plot(simulation.Map[0, :], simulation.Map[1, :], "*k")
    ax1.plot(hxTrue[0, :], hxTrue[1, :], "-k", label="True")
    if iFeature != None: ax1.plot([simulation.xTrue[0][0], simulation.Map[0, iFeature]], [simulation.xTrue[1][0], simulation.Map[1, iFeature]], "-b")

    # Plot odometry trajectory
    ax1.plot(hxOdom[0, :], hxOdom[1, :], "-g", label="Odom")

    # Plot estimated trajectory and current particles
    ax1.plot(hxEst[0, :], hxEst[1, :], "-r", label="Part. Filt.")  
    ax1.plot(xEst[0], xEst[1], ".r")
    
    ax1.scatter(xParticles[0, :], xParticles[1, :], s=wp*10)
    for i in range(nParticles):
        ax1.arrow(xParticles[0, i], xParticles[1, i], 5*np.cos(xParticles[2, i]+np.pi/2), 5*np.sin(xParticles[2, i]+np.pi/2), color = 'orange')

    ax1.axis([-48, 10, -51, -17])
    ax1.grid(True)
    ax1.legend()

    # plot errors curves
    ax3.plot(hxError[0, :], 'b')
    ax3.plot( 3.0 * hxSTD[0, :], 'r')
    ax3.plot(- 3.0 * hxSTD[0, :], 'r')
    ax3.grid(True)
    ax3.set_ylabel('x')
    ax3.set_title('Real error (blue) and particles covariances (red)')


    ax4.plot(hxError[1, :], 'b')
    ax4.plot( 3.0 * hxSTD[1, :], 'r')
    ax4.plot(- 3.0 * hxSTD[1, :], 'r')
    ax4.grid(True)
    ax4.set_ylabel('y')

    ax5.plot(hxError[2, :], 'b')
    ax5.plot( 3.0 * hxSTD[2, :], 'r')
    ax5.plot(- 3.0 * hxSTD[2, :], 'r')
    ax5.grid(True)
    ax5.set_ylabel(r"$\theta$")

    if save: plt.savefig(r'outputs/SRL' + str(k) + '.png')
#        plt.pause(0.01)


# =============================================================================
# Main Program
# =============================================================================

# Enable/disable plotting
is_plot = True

# Nb of particle in the filter
nParticles = 300

# Simulation time
Tf = 1000      # final time (s)
dt_pred = 1     # Time between two dynamical predictions (s)
dt_meas = 1     # Time between two measurement updates (s)

# Location of landmarks
nLandmarks = 5
Map = 120*np.random.rand(2, nLandmarks)-60


# True covariance of errors used for simulating robot movements
QTrue = np.diag([0.02, 0.02, 1*pi/180]) ** 2
RTrue = np.diag([0.5, 1*pi/180]) ** 2

# Modeled errors used in the Particle filter process

QEst = 2 * np.eye(3, 3) @ QTrue 
REst = 2 * np.eye(2, 2) @ RTrue 

# initial conditions
xTrue = np.array([[1, -50, 0]]).T
#xTrue = np.array([[1, -40, -pi/2]]).T
xOdom = xTrue

# initial conditions: - a point cloud around truth
xParticles = xTrue + np.diag([1, 1, 0.1]) @ np.random.randn(3, nParticles)

# initial conditions: global localization
#xParticles = 120 * np.random.rand(3, nParticles)-60

# initial weights
wp = np.ones((nParticles))/nParticles
wp = wp / np.sum(wp)

# initial estimate
xEst = np.average(xParticles, axis=1, weights=wp)
xEst = np.expand_dims(xEst, axis=1)
xSTD = np.sqrt(np.average((xParticles-xEst)*(xParticles-xEst),
               axis=1, weights=wp))
xSTD = np.expand_dims(xSTD, axis=1)

# Init history matrixes
hxEst = xEst
hxTrue = xTrue
hxOdom = xOdom
err = xEst - xTrue
err[2, 0] = angle_wrap(err[2, 0])
hxError = err
hxSTD = xSTD

count = 0
# Simulation environment
simulation = Simulation(Tf, dt_pred, xTrue, QTrue, xOdom, Map, RTrue, dt_meas)

if is_plot: plotParticles(simulation, 0, None, hxTrue, hxOdom, hxEst, hxError, hxSTD, save = True)

histograma = []
cmap=mpl.colormaps["Wistia"]

# Temporal loop
for k in range(1, simulation.nSteps):

    # Simulate robot motion
    simulation.simulate_world(k)

    #Print process
    if (k*100)/simulation.nSteps%1==0: 
        print("Proceso: ", int((k*100)/simulation.nSteps), "%")

    # Get odometry measurements
    xOdom, u_tilde = simulation.get_odometry(k)

    xPartPred=np.zeros((3, nParticles)) 
    
    for p in range(nParticles): 
        noise = np.random.multivariate_normal([0, 0, 0], QEst) 
        noise = np.expand_dims(noise, axis=1) 

        # Prediction: for each particle we add control vector AND noise
        xPartPred[:,p] = np.squeeze(motion_model(xParticles[:,p], u_tilde+noise, dt_pred)) 

    # observe a random feature
    [z, iFeature] = simulation.get_observation(k)

    if z is not None:
        for p in range(nParticles):
            # Predict observation from the particle position
            zPred = observation_model(xPartPred[:,p], iFeature, Map) 
            zPred = np.array([zPred]).T 

            # Innovation : perception error
            Innov = z - zPred 
            Innov[1,0] = angle_wrap(Innov[1,0]) 

            # Compute particle weight using gaussian model
            wp[p]=wp[p]*np.exp(-0.5*Innov.T @ np.linalg.inv(REst) @ Innov)

    # Normalization
    wp = wp/np.sum(wp) 

    # Compute position as weighted mean of particles
    xEst = np.average(xPartPred, axis=1, weights=wp) 
    xEst = np.expand_dims(xEst, axis=1)

    xSTD = np.sqrt(np.average((xPartPred-xEst)*(xPartPred-xEst), axis=1, weights=wp)) 
    xSTD = np.expand_dims(xSTD, axis=1)

    # Multinomial resampling
    theta_eff = 0.1
    Nth = nParticles * theta_eff #Threshold for resampling (0.1 times of the number of particles)
    Neff = 1 / np.sum(np.square(wp)) #Effective sample size
    
    if Neff < Nth: #if the Neff is below a predetermined number of samples
        # Particle resampling
        xParticles, wp = re_sampling_sys(xPartPred, wp) #removes low weighted particles and selects particles with high weight to reduce errors due to probability distributions.
        
    else:
        xParticles = xPartPred


    # store data history
    hxTrue = np.hstack((hxTrue, simulation.xTrue))
    hxOdom = np.hstack((hxOdom, simulation.xOdom))
    hxEst = np.hstack((hxEst, xEst))
    err = xEst - simulation.xTrue
    err[2, 0] = angle_wrap(err[2, 0])
    hxError = np.hstack((hxError, err))
    hxSTD = np.hstack((hxSTD, xSTD))

    # plot simulation curves every 20 updates 
    if is_plot and k*simulation.dt_pred % 20 == 0:
        plotParticles(simulation, k, iFeature, hxTrue, hxOdom, hxEst, hxError, hxSTD, save = True)

    # plot histogram every 100 updates
    if is_plot and k*simulation.dt_pred % 100 == 0:

        hist, bins = np.histogram(wp, bins=100, range=(0.0, 0.05))
        histograma.append(hist)
        
plt.figure()

for i, hist in enumerate(histograma):
    plt.fill_between(bins[:-1], hist, alpha=(0.1+0.3*hist/np.max(histograma)), color = cmap(i*25))

# Set the title and labels
plt.title("Histogramme superposé des poids")
plt.xlabel("Poids")
plt.ylabel("Nombre de particules")
plt.legend()  # Show a legend with iteration numbers

# Set the colorbar
norm = mpl.colors.Normalize(vmin=1, vmax=1000) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
sm.set_array([]) 
plt.colorbar(sm, ticks=np.linspace(1, 1000, 11)) 

plt.show()

tErrors = np.sqrt(np.square(hxError[0, :]) + np.square(hxError[1, :]))
print("Mean (var) translation error : {:e} ({:e})".format(np.mean(tErrors), np.var(tErrors)))
print("Press Q in figure to finish...")
plt.show()