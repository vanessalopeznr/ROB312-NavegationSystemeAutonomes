"""
TP Kalman filter for mobile robots localization

authors: Goran Frehse, David Filliat, Nicolas Merlinge
"""

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


    # generate a noisy observation of a random feature(Modelo no linear)
    #Yt=H*Xt+Wt
    #H = measurement matrix (used to convert the state into predicted sensor observations)
    #Xt = state vector
    #Wt = noise of each of sensor observations
    def get_observation(self, k):
        # Ensuring random repetability for given k
        np.random.seed(seed*3 + k)

        # Model 
        if k*self.dt_pred % self.dt_meas == 0:
            notValidCondition = False # False: measurement valid / True: measurement not valid
            #notValidCondition = False if k < 2500 or k > 3500 else True
            if notValidCondition:
                z = None
                iFeature = None
            else:
                iFeature = np.random.randint(0, self.Map.shape[1] - 1)
                zNoise = np.sqrt(self.RTrue) @ np.random.randn(2)
                zNoise = np.array([zNoise]).T
                z = observation_model(self.xTrue, iFeature, self.Map) + zNoise.T
                z=z.T
                z[1, 0] = angle_wrap(z[1, 0])
        else:
            z = None
            iFeature = None
        return [z, iFeature]


# ---- Kalman Filter: model functions ----

# evolution model (f)
def motion_model(x, u, dt_pred):
    # xPred: La posicion actual = posicion anterior + velocidad*tiempo (diap 27 - Xk)
    # x: estimated state (x, y, heading)
    # u: control input or odometry measurement in body frame (Vx, Vy, angular rate)
    xPred=tcomp(x,u,dt_pred)
    
    return xPred

# observation model (h):  we use the state to estimate sensor measurements that represents a vector of predicted sensor measurements y
def observation_model(xVeh, iFeature, Map):
    #One landmark is randomly selected to simulate a measurement
    # xVeh: vecule state
    # iFeature: observed amer index (landmark index)
    # Map: map of all amers (2x30)
    xP=Map[:,iFeature]
    xVeh=np.squeeze(xVeh)
    z=np.array([sqrt((xP[0]-xVeh[0])**2+(xP[1]-xVeh[1])**2),atan2((xP[1]-xVeh[1]),(xP[0]-xVeh[0]))-xVeh[2]])
    return z


# ---- Kalman Filter: Jacobian functions to be completed ----

# h(x) Jacobian wrt x
def  get_obs_jac(xPred, iFeature, Map):
    # xPred: predicted state
    # iFeature: observed amer index
    # Map: map of all amers
    xP=Map[:,iFeature]
    xPred=np.squeeze(xPred)
    H=np.array([[-(xP[0]-xPred[0])/sqrt((xP[0]-xPred[0])**2+(xP[1]-xPred[1])**2), -(xP[1]-xPred[1])/sqrt((xP[0]-xPred[0])**2+(xP[1]-xPred[1])**2), 0],
                [(xP[1]-xPred[1])/((xP[0]-xPred[0])**2+(xP[1]-xPred[1])**2), -(xP[0]-xPred[0])/((xP[0]-xPred[0])**2+(xP[1]-xPred[1])**2), -1]])
    return H


# f(x,u) Jacobian wrt x
def F(x, u, dt_pred):
    # x: estimated state (x, y, heading)
    # u: control input (Vx, Vy, angular rate)
    # dt_pred: time step
    #Primera columna es la derivada del modelo Xk con respecto a x, la segnda con respecto a y y la tercera con respecto a theta
    x=np.squeeze(x)
    u=np.squeeze(u)
    Jac=np.array([[1, 0, -u[0]*sin(x[2])*dt_pred-u[1]*cos(x[2])*dt_pred],
                  [0, 1, u[0]*cos(x[2])*dt_pred-u[1]*sin(x[2])*dt_pred],
                  [0, 0, 1]])
    return Jac


# f(x,u) Jacobian wrt w (noise on the control input u)
def G(x, u, dt_pred):
    # x: estimated state (x, y, heading) in ground frame
    # u: control input (Vx, Vy, angular rate) in robot frame
    # dt_pred: time step for prediction 
    Jac=np.array([[cos(x[2])*dt_pred, -sin(x[2])*dt_pred, 0],
                  [sin(x[2])*dt_pred, cos(x[2])*dt_pred, 0],
                  [0, 0, dt_pred]])
    return Jac


# ---- Utils functions ----
# Display error ellipses
def plot_covariance_ellipse(xEst, PEst, axes, lineType):
    """
    Plot one covariance ellipse from covariance matrix
    """

    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    if eigval[smallind] < 0:
        print('Pb with Pxy :\n', Pxy)
        exit()

    t = np.arange(0, 2 * pi + 0.1, 0.1)
    a = sqrt(eigval[bigind])
    b = sqrt(eigval[smallind])
    x = [3 * a * cos(it) for it in t]
    y = [3 * b * sin(it) for it in t]
    angle = atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    rot = np.array([[cos(angle), sin(angle)],
                    [-sin(angle), cos(angle)]])
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    axes.plot(px, py, lineType)


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


# =============================================================================
# Main Program
# =============================================================================

# Init displays
show_animation = True
save = False

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
ax3 = plt.subplot(3, 2, 2)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 6)

# ---- General variables ----

# Simulation time
Tf = 6000       # final time (s)
dt_pred = 1     # Time between two dynamical predictions (s)
dt_meas = 1    # Time between two measurement updates (s)

# Location of landmarks
nLandmarks = 30
Map = 140*(np.random.rand(2, nLandmarks) - 1/2)

# True covariance of errors used for simulating robot movements
# Datos reales a los cuales el filtro de kalman no tiene acceso
QTrue = np.diag([0.01, 0.01, 1*pi/180]) ** 2
RTrue = np.diag([5.0, 6*pi/180]) ** 2

# Modeled errors used in the Kalman filter process
# Datos que tiene acceso el filtro de kalman
QEst = 100*np.eye(3, 3) @ QTrue
REst = 10*np.eye(2, 2) @ RTrue

# initial conditions
xTrue = np.array([[1, -40, -pi/2]]).T
xOdom = xTrue
xEst = xTrue
print("Initial state: ", xEst)
PEst = 10 * np.diag([1, 1, (1*pi/180)**2])

# Init history matrixes
hxEst = xEst
hxTrue = xTrue
hxOdom = xOdom
hxError = np.abs(xEst-xTrue)  # pose error
hxVar = np.sqrt(np.diag(PEst).reshape(3, 1))  # state std dev
htime = [0]

# Simulation environment
simulation = Simulation(Tf, dt_pred, xTrue, QTrue, xOdom, Map, RTrue, dt_meas)

# Temporal loop
for k in range(1, simulation.nSteps):
        
    # Simulate robot motion
    simulation.simulate_world(k)

    # Get odometry measurements
    xOdom, u_tilde = simulation.get_odometry(k)

    # Kalman prediction
    xPred = motion_model(xEst,u_tilde,dt_pred)  # function f
    Fmat=F(xEst,u_tilde,dt_pred)
    Gmat=G(xEst,u_tilde,dt_pred)
    PPred = (np.dot(Fmat,np.dot(PEst,Fmat.T)))+np.dot(Gmat,np.dot(QEst,Gmat.T))

    if k==1 or k % 1000 == 0 :
        print("Iteration: ", k)


    # Get random landmark observation
    [z, iFeature] = simulation.get_observation(k) #z: actual readings from the sensors (Yt)

    if z is not None:
    #if False:
        # Predict observation
        zPred = observation_model(xPred, iFeature, Map) # predicted sensor measurements given the predicted state estimate 
        zPred=zPred.reshape(2,1)

        # get observation Jacobian
        H = get_obs_jac(xPred, iFeature, Map)

        only = "none"

        if only == 'range':
            Innov=z[0]-zPred[0]   
            Innov = angle_wrap(Innov)
            Innov=np.squeeze(Innov)
            H = H[0, :]
            H=np.array([H])
            S = REst[0,0]+np.dot(H,np.dot(PPred,H.T))
            K = np.dot(PPred,np.dot(H.T,1/S))

        elif only == 'angle':
            Innov=z[1]-zPred[1]   
            Innov = angle_wrap(Innov)
            Innov=np.squeeze(Innov)
            H = H[1, :]
            H=np.array([H])
            S = REst[1,1]+np.dot(H,np.dot(PPred,H.T))
            K = np.dot(PPred,np.dot(H.T,1/S))

        else:
            # compute Kalman gain - with dir and distance
            Innov = z-zPred   # observation error (innovation)
            Innov[1, 0] = angle_wrap(Innov[1, 0])
            S = REst + np.dot(H,np.dot(PPred,H.T))
            K = np.dot(PPred,np.dot(H.T,np.linalg.inv(S)))

        # Compute Kalman gain to use only distance 
#        Innov =        # observation error (innovation)
#        H = #...................
#        S = #...................
#        K = #...................

        # Compute Kalman gain to use only direction
#        Innov = #...................       # observation error (innovation)
#        Innov[1, 0] = angle_wrap(Innov[1, 0])
#        H = #...................           # observation error (innovation)
#        S = #...................
#        K = #...................
        
        # perform kalman update
        xEst = xPred + np.dot(K,Innov)
        xEst[2, 0] = angle_wrap(xEst[2, 0])

        PEst = np.dot(np.eye(3)-np.dot(K,H),PPred)
        
        
        PEst = 0.5 * (PEst + PEst.T)  # ensure symetry
        
    else:
        # there was no observation available
        xEst = xPred
        PEst = PPred

    # store data history
    hxTrue = np.hstack((hxTrue, simulation.xTrue))
    hxOdom = np.hstack((hxOdom, simulation.xOdom))
    hxEst = np.hstack((hxEst, xEst))
    err = xEst - simulation.xTrue
    err[2, 0] = angle_wrap(err[2, 0])
    hxError = np.hstack((hxError, err))
    hxVar = np.hstack((hxVar, np.sqrt(np.diag(PEst).reshape(3, 1))))
    htime.append(k*simulation.dt_pred)
    
    # plot every 15 updates
    if show_animation and k*simulation.dt_pred % 200 == 0:
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

        ax1.cla()
        
        times = np.stack(htime)
        
        # Plot true landmark and trajectory
        ax1.plot(Map[0, :], Map[1, :], "*k")
        ax1.plot(hxTrue[0, :], hxTrue[1, :], "-k", label="True")

        # Plot odometry trajectory
        ax1.plot(hxOdom[0, :], hxOdom[1, :], "-g", label="Odom")

        # Plot estimated trajectory an pose covariance
        ax1.plot(hxEst[0, :], hxEst[1, :], "-r", label="EKF")
        ax1.plot(xEst[0], xEst[1], ".r")
        plot_covariance_ellipse(xEst,
                                PEst, ax1, "--r")

        ax1.axis([-70, 70, -70, 70])
        ax1.grid(True)
        ax1.legend()
        
        # plot errors curves
        ax3.plot(times, hxError[0, :], 'b')
        ax3.plot(times, 3.0 * hxVar[0, :], 'r')
        ax3.plot(times, -3.0 * hxVar[0, :], 'r')
        ax3.grid(True)
        ax3.set_ylabel('x')
        ax3.set_xlabel('time (s)')
        ax3.set_title('Real error (blue) and 3 $\sigma$ covariances (red)')
        
        ax4.plot(times, hxError[1, :], 'b')
        ax4.plot(times, 3.0 * hxVar[1, :], 'r')
        ax4.plot(times, -3.0 * hxVar[1, :], 'r')
        ax4.grid(True)
        ax4.set_ylabel('y')
        ax5.set_xlabel('time (s)')

        ax5.plot(times, hxError[2, :], 'b')
        ax5.plot(times, 3.0 * hxVar[2, :], 'r')
        ax5.plot(times, -3.0 * hxVar[2, :], 'r')
        ax5.grid(True)
        ax5.set_ylabel(r"$\theta$")
        ax5.set_xlabel('time (s)')
        
        if save: plt.savefig(r'outputs/EKF_' + str(k) + '.png')
#        plt.pause(0.001)

plt.savefig('EKFLocalization.png')
print("Press Q in figure to finish...")
plt.show()

