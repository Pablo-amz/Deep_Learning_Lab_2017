# Hyperparameter optimization using surrogates
# Exercise 5 of the Deep Learning Lab, ML track
# Pablo de Andres, WS 2017/18

import ho_exercise as ho
import math
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from robo.fmin import bayesian_optimization

# Values for number of runs and iterations
RUNS = 10
ITERATIONS = 50
class HyperparameterOptimization:
    """Class that implements both hyperparameter optimization methods"""
    def __init__(self):
        """ Initialises the bounds of the hyperparameters. """ 
        self._learning_rate = [-6, 0]
        self._batch_size = [35, 512]
        self._number_filters = [4, 10]
        self.rs_performance = np.zeros(shape = (RUNS, ITERATIONS))
        self.rs_runtime = np.zeros(shape = (RUNS, ITERATIONS))
        self.bo_performance = []
        self.bo_runtime = []
    
    def random_search(self):
        """ Implementation of random search"""
        # Initial value for the error
        best = 1
        for i in range(RUNS):
            for j in range(ITERATIONS):
                # Set the parameters for the random search
                rate = rd.randint(self._learning_rate[0],
                                  self._learning_rate[1])
                batch = rd.randint(self._batch_size[0], self._batch_size[1])
                filters1 = rd.randint(self._number_filters[0],
                                       self._number_filters[1])
                filters2 = rd.randint(self._number_filters[0],
                                      self._number_filters[1])
                filters3 = rd.randint(self._number_filters[0],
                                      self._number_filters[1])
                input_x = [rate, batch, filters1, filters2, filters3]
                error = ho.objective_function(input_x)
                self.rs_runtime[i][j] = ho.runtime(input_x)
                # If it's an improvement
                if error < best:
                    self.rs_performance[i][j] = error
                    best = error
                else:
                    self.rs_performance[i][j] = best
        # Mean over all the runs
        self.rs_performance = np.mean(self.rs_performance, axis=0)
        # Cumulative sum of the mean for all the runs
        self.rs_runtime = np.cumsum(np.mean(self.rs_runtime, axis=0), axis=0)
        
        
    def bayesian_optimization(self):
        """ Implementation of the Bayesian Optimization. """
        # Set the parameters for the bayesian optimization
        function = ho.objective_function
        lower = np.array([self._learning_rate[0], self._batch_size[0], 
                          self._number_filters[0], self._number_filters[0],
                          self._number_filters[0]])
        upper = np.array([self._learning_rate[1], self._batch_size[1], 
                          self._number_filters[1], self._number_filters[1],
                          self._number_filters[1]])
        for i in range(RUNS):
            results = bayesian_optimization(function, lower, upper,
                                            num_iterations=ITERATIONS)
            self.bo_performance.append(results['incumbent_values'])
            self.bo_runtime.append(results['runtime'])
        # Mean over all the runs
        self.bo_performance = np.mean(self.bo_performance, axis=0)
        # Cumulative sum of the mean for all the runs
        self.bo_runtime = np.cumsum(np.mean(self.bo_runtime, axis=0), axis=0)
        

    def plot(self, array1, label1="", array2 = None, label2="", xlabel="x",
             ylabel="y", title=""):
        """ Plots the given array."""
        plt.plot(range(ITERATIONS), array1, label=label1)
        if array2 is not None:
            plt.plot(range(ITERATIONS), array2, label=label2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
        
        
        
if __name__ == "__main__":
    h = HyperparameterOptimization()
    h.random_search()
    h.bayesian_optimization()
    # Plots
    h.plot(h.rs_performance, label1="random search", xlabel="iteration",
           ylabel="mean values", title="Random search")
    h.plot(h.rs_performance, "random", h.bo_performance, "bayesian", 
           "iteration", "mean values", "Random search vs. Bayesian optimization")
    h.plot(h.rs_runtime, "random", h.bo_runtime, "bayesian", "iteration",
           "mean values", "Runtime comparison")
