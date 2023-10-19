import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def classify_states(prices: np.array, states: np.array, n_states: int) -> dict:
    """
    Classify states based on DTW distances to standard patterns 
    and plot their cumulative returns.
    
    Parameters:
    - prices: A numpy array of prices.
    - states: A numpy array of hidden states.
    - n_states: Number of unique states.
    
    Returns:
    - A dictionary with state classifications.
    """

    def save_classification(state_classifications: dict) -> None:

        """
        Save the classified states in a dataframe format so that
        predictions could access it without needing to do the whole
        calculation over again

        Parameters:
        - state_classifications: a dictionary containing states map

        Returns:
        - None. File is saved in the specified directory.
        """

        df = pd.DataFrame([state_classifications])
        df.to_csv('Models/state_classification.csv', index=False)
    
    def compute_dtw(sequence, pattern):
        """Compute DTW distance between two sequences."""
        n, m = len(sequence), len(pattern)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, 1:] = float('inf')
        dtw_matrix[1:, 0] = float('inf')

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(sequence[i-1] - pattern[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

        return dtw_matrix[n, m]
    
    # Calculate returns and cumulative returns for each state
    returns = np.diff(prices, prepend=prices[0])
    cumulative_returns = np.zeros_like(prices)
    for state in range(n_states):
        mask = states == state
        cumulative_returns[mask] = np.cumsum(returns[mask])
    
    # Generate the three standard patterns for DTW
    neutral_pattern = np.zeros(len(prices))
    bear_pattern = np.linspace(0, -1, len(prices))
    bull_pattern = np.linspace(0, 1, len(prices))
    
    patterns = {
        'yellow': neutral_pattern,
        'red': bear_pattern,
        'green': bull_pattern
    }

    # Classify states based on DTW distances to the patterns using the custom DTW implementation
    state_classifications = {}

    # Set up the subplots
    nrows = (n_states + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for state in range(n_states):
        state_data = cumulative_returns[states == state]
        
        # Compute DTW distances for the state to each pattern
        distances = {name: compute_dtw(state_data, pattern) for name, pattern in patterns.items()}
        
        # Classify state based on the pattern with minimum DTW distance
        classification = min(distances, key=distances.get)
        state_classifications[state] = classification

        # Plotting the cumulative returns for the state
        ax = axes[state]
        ax.plot(state_data, label=f'State {state} (Classified as {classification})')
        ax.legend()
        ax.set_title(f'State {state}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Returns')
        
    # Adjust layout
    plt.tight_layout()
    plt.show()

    save_classification(state_classifications)

    return state_classifications


"""
np.random.seed(0)
sample_prices = np.random.rand(100)
sample_states = np.random.randint(0, 8, size=100)

classify_states(sample_prices, sample_states, 8)
"""