import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import matplotlib.gridspec as gridspec


def GenerateJDP(NoOfPaths, NoOfSteps, S_0, T, Drift, var, Int, Amp, Z):

    """
    Generates paths for a Jump Diffusion Process (JDP) model.
    
    Parameters:
        - NoOfPaths: Number of paths to simulate.
        - NoOfSteps: Number of discrete steps.
        - S_0: Initial price.
        - T: Time period.
        - Drift: Drift rate.
        - var: Volatility.
        - Int: Jump Intensity.
        - Amp: Jump Amplitude.
        - Z: Normally distributed random numbers.

    Formula:
        X[t+1] = X[t] * (1 + Amp * jumps*dt + Drift*dt + var*Z*np.sqrt(dt))
    
    Returns:
        Array of JDP model paths.
    """

    jump_signs = np.random.choice([1, -1], Z.shape)
    jumps = jump_signs * np.random.poisson(Int, Z.shape)
    
    X = S_0 * np.ones((NoOfPaths, NoOfSteps + 1))
    dt = T / NoOfSteps
    
    for i in range(0, NoOfSteps):
        #                          [-- Modelling Jumps --] [Drift part]  [---- Stochastic Component ----]
        X[:, i+1] = X[:, i] * (1 + Amp * jumps[:, i] * dt + Drift * dt + var * Z[:, i] * np.sqrt(dt))
    return X


def GeneratePathsOU(NoOfPaths, NoOfSteps, T, sigma, theta, S_0, Z, mean):

    """
    Generates paths for the Ornstein-Uhlenbeck (OU) model.
    
    Parameters:
        - NoOfPaths: Number of paths to simulate.
        - NoOfSteps: Number of discrete steps.
        - T: Time period.
        - sigma: Volatility.
        - theta: Rate at which the process reverts to the mean.
        - S_0: Initial price.
        - Z: Normally distributed random numbers.

    Formula:
        X[t+1] = X[t] + theta * (S_0 - X[t]) * dt + sigma*dW[t]
    
    Returns:
        Dictionary containing time and simulated paths.
    """
    
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])

    X[:, 0] = S_0  # Start with the initial price, not its log

    dt = T / float(NoOfSteps)
    dW = np.sqrt(dt) * Z

    for t in range(0, NoOfSteps):

        if NoOfPaths > 1:
            # Making sure that samples from normal have mean 0 and variance 1
            # Essentially simply standartizing the data
            Z[:,t] = (Z[:,t] - np.mean(Z[:,t])) / np.std(Z[:,t])

        #                    [- Deterministic part -]  [- Stochastic Part -]
        X[:, t+1] = X[:, t] + theta * (mean - X[:, t]) * dt + sigma * dW[:, t]
        time[t+1] = time[t] + dt

    paths = {"time":time,"S":X}
    return paths


def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0, Z):    

    """
    Generates paths for the Brownian Motion (BM) model.
    
    Parameters:
        - NoOfPaths: Number of paths to simulate.
        - NoOfSteps: Number of discrete steps.
        - T: Time period.
        - r: Risk-free rate.
        - sigma: Volatility.
        - S_0: Initial price.
        - Z: Normally distributed random numbers.

    Formula:
        X[t+1] = X[t] + (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z[t]
        S[t] = exp(X[t])
    
    Returns:
        Dictionary containing time and simulated paths.
    """

    # Creating a random np array (Z) normally distributted and empty lists
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])   

    # Fill the initial row with log value of starting position
    X[:,0] = np.log(S_0)

    # Change period
    dt = T / float(NoOfSteps)

    for i in range(0,NoOfSteps):
        if NoOfPaths > 1:
            # Making sure that samples from normal have mean 0 and variance 1
            # Essentially simply standartizing the data
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])

        #                   [------ Drift Component ------]  [----- Volatility Component -----]
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma * sigma) * dt + sigma * np.power(dt, 0.5) * Z[:,i]
        #                                                                           [ Randomness (Z) ]
        
        time[i+1] = time[i] +dt
        
    #Compute exponent of ABM
    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths


def MainCode():

    spy = yf.download("SPY", start="2023-01-01", progress=False)
    
    spy_close     = np.array(spy['Close'])[:-5] # 5 days back
    spy_current   = np.array(spy['Close'])[-1]
    daily_returns = np.diff(spy_close) / spy_close[:-1]
    spy_ma_100    = np.mean(spy_close[-100:])

    NoOfPaths     = 1000                  # Number of paths
    NoOfSteps     = 32                    # Number of steps 
    S_0           = spy_close[-1]         # Initial price
    r             = 0.001                 # Risk free rate
    mu            = 0                     # Expected return
    sigma         = np.std(daily_returns) # Volatility
    T             = 5                     # Time period of 5 days
    sims          = 3                     # Simulations qty
    theta         = 0.5                   # How quickly reverts to the mean (1 = daily)
    JumpIntensity = 0.03                  # Jump intensity per T
    JumpAmplitude = 0.1                   # Jump amplitude
    mean          = spy_ma_100            # Mean


    np.random.seed(595955)
    Z_single = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    Z = np.tile(Z_single, (sims,1,1))
    
    # Paths simulation
    pathsr = GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0, Z[0, :, :])
    time= pathsr["time"] 
    pathsr = pathsr["S"]
    pathsou = GeneratePathsOU(NoOfPaths, NoOfSteps, T, sigma*1000, theta, S_0, Z[1, :, :], mean)["S"]
    pathsjdp = GenerateJDP(NoOfPaths, NoOfSteps, S_0, T, mu, sigma, JumpIntensity, JumpAmplitude, Z[2,:,:])

    # Plotting single lines to visualize differences
    fix1 = plt.figure(1)
    plt.plot(time, np.transpose(pathsr[:1, :]),'green', label="BM")
    plt.plot(time, np.transpose(pathsou[:1, :]),'red', label="OU")
    plt.plot(time, np.transpose(pathsjdp[:1, :]),'orange', label="JDP")
    plt.axhline(S_0, color='blue', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.show()

    # Plotting all histograms in one place to visualize differences
    fig2 = plt.figure(2)
    plt.hist(pathsr[:, -1], bins=50, alpha=0.5, label="GBM r", color="orange")
    plt.axvline(np.mean(pathsr), color='orange', linestyle='dashed', linewidth=1)
    plt.hist(pathsou[:, -1], bins=50, alpha=0.5, label="OU", color="black")
    plt.axvline(np.mean(pathsou), color='black', linestyle='dashed', linewidth=1)
    plt.hist(pathsjdp[:, -1], bins=50, alpha=0.5, label="JDP", color="grey")
    plt.axvline(np.mean(pathsjdp), color='grey', linestyle='dashed', linewidth=1)
    plt.title("Distribution of final prices")
    plt.xlabel("Final Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Function to plot individual figures
    def plot_individual_figure(method_name, prices):
        fig = plt.figure(figsize=(15, 7))
        gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])  # width_ratios adjusted

        # Path chart
        ax0 = plt.subplot(gs[0])
        ax0.plot(spy_close, color="blue", label=method_name)
        ax0.set_title(f"{method_name} Histogram on SPY")
        ax0.set_ylabel("Price")
        ax0.set_xlabel("Time")
        ax0.scatter(len(spy_close)+4, spy_current, marker="X", label="Current Price", color="red")
        ax0.margins(x=0)
        ax0.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)

        # Histogram
        ax1 = plt.subplot(gs[1])
        hist, bins = np.histogram(prices[:, -1], bins=50, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        ax1.barh(bin_centers, hist, height=(bins[1]-bins[0]), color="blue", align='center')
        ax1.axhline(np.mean(prices[:, -1]), color='black', linestyle='--', label="Mean")
        ax1.set_ylim(ax0.get_ylim())  # Setting y limits of histogram to match ax0
        ax1.axhline(np.mean(prices[:, -1]) + 2*np.std(prices), color='green', linestyle='--', label="98% Confidence interval")
        ax1.axhline(np.mean(prices[:, -1]) - 2*np.std(prices), color='green', linestyle='--')
        ax1.yaxis.set_visible(False)

        plt.subplots_adjust(wspace=0)
        plt.legend()

    # # Plot for AMB
    plot_individual_figure("ABM", pathsr)

    # # Plot for OU
    plot_individual_figure("OU", pathsou)

    # Plot for JDP
    plot_individual_figure("JDP", pathsjdp)

    plt.show()

MainCode()
