from Models.HMM_trainer import HMMTrainer
import numpy as np
import logging

class HMMCalculatorBIC:
    
    """
    BIC Calculator for Hidden Markov Models (HMM).

    This class provides functionality to compute the Bayesian Information Criterion (BIC) 
    for various numbers of components in HMM. 
    
    It supports multiple iterations for each component count and can return either 
    average BIC values across iterations or the entire array of BIC values.

    Attributes:
        BIC_iterations (int): Number of iterations for each component count.
        BIC_max_components (int): Maximum number of components to consider.
        BICs (np.array): array storing BIC values for each iteration and component.
    
    """
    def __init__(self, BIC_iterations: int, BIC_max_components: int) -> None:
        
        """
        Initializes the BIC_Calculator_HMM with the specified 
        number of iterations and maximum components.

        Args:
            BIC_iterations (int): Number of iterations for each component count.
            BIC_max_components (int): Maximum number of components to consider.
        """

        self._BIC_iterations     = BIC_iterations
        self._BIC_max_components = BIC_max_components
        self._BICs               = np.zeros((BIC_iterations, BIC_max_components))
        self._logger             = logging.getLogger(__name__)

    def calculate(self, data: np.array, return_average: bool = True, callback: callable = None) -> np.array:
        
        """
        Calculate the BIC values for the given data and specified range of components.

        For each number of components (from 1 to BIC_max_components), 
        an HMM is trained, and the BIC is computed. 
        This process is repeated for the specified number of iterations.

        Args:
            data (np.array): Input data array.
            return_average (bool, optional): If True, returns the average BIC values. 
                                             If False, returns array of BIC values.

        Returns:
            np.array: Either the average BIC values or the array of BIC values.
        """

        if not isinstance(data, np.ndarray):
            raise ValueError("Data should be a numpy array.")
        
        for j in range(self._BIC_iterations):
            self._logger.info(f"Iteration: {j}")

            for n in range(1, self._BIC_max_components):
                self._logger.debug(f"Component iteration: {n}")

                try:
                    HMM = HMMTrainer(data).train_model(n)
                except Exception as e:
                    self._logger.error(f"Error training HMM for {n} components: {e}")
                    continue

                k                  = n*(n-1) + n + n
                log_likelihood     = HMM.score(data)
                self._BICs[j, n-1] = -2.0 * log_likelihood  + k * np.log(len(data))
                
                if callback:
                    callback(self._BICs[j, n-1])

        if return_average:
            return np.mean(self._BICs, axis=0)
        else:
            return self._BICs






