import copy
from types import SimpleNamespace

import numpy as np
import scipy.optimize
from pycosat import solve
from scipy import optimize
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.linalg import lstsq as sp_lstsq
from scipy.optimize import brentq


class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5

        # c. household production
        par.alpha = 0.5
        par.sigma = 1

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8, 1.2, 5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

        #Added parameter to question 5:
        par.theta = 1
        par.gamma = 1

    def calc_utility(self, LM, HM, LF, HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM * LM + par.wF * LF

        # b. home production
        if par.sigma == 1:
            H = HM ** (1 - par.alpha) * HF ** par.alpha
        elif par.sigma == 0:
            H = min(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha * HF ** ((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C ** par.omega * H ** (1 - par.omega)
        utility = np.fmax(Q, 1e-8) ** (1 - par.rho) / (1 - par.rho)

        # d. disutility  of work
        epsilon_ = 1 + 1 / par.epsilon
        TM = LM + HM
        TF = LF + HF
        disutility = par.nu * (TM ** epsilon_ / epsilon_) + (par.nu**par.theta) * (TF ** epsilon_ / epsilon_)
        #Changed the disutility in question 5


        return utility - disutility
    def solve_discrete(self, do_print=False):
        """ solve model discretely """

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. all possible choices
        x = np.linspace(0, 24, 49)
        LM, HM, LF, HF = np.meshgrid(x, x, x, x)  # all combinations

        LM = LM.ravel()  # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM, HM, LF, HF)

        # c. set to minus infinity if constraint is broken
        I = (LM + HM > 24) | (LF + HF > 24)  # | is "or"
        u[I] = -np.inf

        # d. find maximizing argument
        j = np.argmax(u)

        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k, v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    def solve(self):
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective_func(x):
            return -self.calc_utility(x[0], x[1], x[2], x[3]) # Utility, minus to maximize.


        # Define the initial guess and bounds
        initial_guess = [12, 12, 12, 12]
        bounds = ((0, 24), (0, 24), (0, 24), (0, 24))

        # Add the constraint to the optimization problem
        constraint1 = lambda x: 24 - (x[0] + x[1])  # LM + HM <= 24
        constraint2 = lambda x: 24 - (x[2] + x[3])  # LF + HF <= 24

        # Use lambda function to pass fixed values of LF and HF to objective function
        sol = optimize.minimize(lambda x: objective_func(x), initial_guess,
                                method='SLSQP', bounds=bounds,
                                constraints=[{'type': 'ineq', 'fun': constraint1},
                                             {'type': 'ineq', 'fun': constraint2}],
                                tol = 1e-9)   # Low tolerance. Increases the likely hood, we have found the correct maximization value.

        opt.LM = sol.x[0]
        opt.HM = sol.x[1]
        opt.LF = sol.x[2]
        opt.HF = sol.x[3]
        opt.u = sol.fun # Utility


        return opt

    def solve_wF_vec(self, discrete=False):
        """ solve model for vector of female wages """


        sol = self.sol
        par = self.par
        opt = SimpleNamespace()

        for i, w_F in enumerate(par.wF_vec):
            par.wF = w_F
            if discrete==True:
                opt = self.solve_discrete()
            elif discrete == False:
                opt = self.solve()

            else:
                print('Error: discrete must be either True / False')

            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

        return sol



    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol


        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec / sol.HM_vec)
        A = np.vstack([np.ones(x.size), x]).T
        sol.beta0, sol.beta1 = np.linalg.lstsq(A, y, rcond=None)[0]


    def estimate(self, alpha=None):
        """ estimate alpha and sigma for variable and fixed alpha """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        if alpha == None:
            def objective_function(x):
                par.alpha = x[0]
                par.sigma = x[1]
                self.solve_wF_vec()
                self.run_regression()
                return (par.beta0_target - sol.beta0) ** 2 + (
                            par.beta1_target - sol.beta1) ** 2

            # Some Bounds and some intial guess. The values are not essentiel, it just take longer to run, if they are less restrictive.
            bnds = [(0,1), (0,1)] #alpha between 0 and 1, sigma between 0 and 1

            initial_guess = [0.99, 0.1]
            # ii. optimizer
            result = optimize.minimize(objective_function, initial_guess,
                                       method='Nelder-Mead',
                                       bounds=bnds,
                                       tol=1e-10,
                                       )

            opt.result = result
            return opt.result




        #This is for question 5..., I select some alpha value and only maximize for sigma.
        else:
            def objective_function(x):
                par.alpha = alpha
                par.sigma = x[1]
                self.solve_wF_vec()
                self.run_regression()
                return (par.beta0_target - sol.beta0) ** 2 + (
                        par.beta1_target - sol.beta1) ** 2

                # Some Bounds and some intial guess. The values are not essentiel, it just take longer to run, if they are less restrictive.

            bnds = [(0.5,0.5), (0,1)]

            initial_guess = [0.5, 0.1]
            # ii. optimizer
            result = optimize.minimize(objective_function, initial_guess,
                                       method='Nelder-Mead',
                                       bounds=bnds,
                                       tol=1e-10,
                                       )

            opt.result = result


            return opt.result

    def estimate2(self):
        """ estimate theta and sigma """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        def objective_function(x):
            par.theta = x[0]
            par.sigma = x[1]
            self.solve_wF_vec()
            self.run_regression()
            return (par.beta0_target - sol.beta0) ** 2 + (
                    par.beta1_target - sol.beta1) ** 2

        bnds = [(0, 5), (0, 10)]
        initial_guess = [2, 1]
        result2 = optimize.minimize(objective_function, initial_guess,
                                    method='Nelder-Mead',
                                    bounds=bnds,
                                    tol = 1e-13,
                                    options={'maxfev': 10000})

        opt.result = result2

        return opt.result










