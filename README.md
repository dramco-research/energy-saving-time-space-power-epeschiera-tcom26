# Optimizing time-, space-, and power-domain energy-saving techniques for sub-6 GHz base stations

<div align="center">
  <a href="https://arxiv.org/abs/2505.15445"><img src="https://img.shields.io/badge/arXiv-%2300629B?style=flat&logo=arXiv" alt="arXiv"></a>
  &nbsp;
</div>
<br>

This repository contains the Python source code to generate Figure 1a-b, Figure 2a-d, and Figure 3a-b, Figure 4a-b, and Figure 5 of the manuscript [1].
<br>If you use this code in your work, please cite our paper as in [1].

The requested Python libraries are: numpy, scipy, and matplotlib.

Each file is structured in: (i) definitions of functions, (ii) initialization of parameters, (iii) core computations, with no looping or looping over one or more variables, and (iv) plot of results.

In Fig_3.py, the base station consumption and number of optimized resources are plotted versus the network load for a single channel realization. Fig_4-5.py extends the analysis to power consumption distribution for the three base station configurations. The median performance in terms of power consumption and energy efficiency versus the number of users is visualized in Fig_6-7.py. The utility functions used across the three files are included in utils.py.


**Note 1:** To select enabled or disabled time-domain hardware power-saving modes, comment or uncomment lines 43 and 44 in any of the .py files.

**Note 2:** The function f(x,y) is defined in equation (23) of the paper. In utils.py, solve_problem0() solves the unconstrained problem min_{x,y} f(x,y). The one-dimensional unconstrained problems are solved by solve_problem1(), which solves min_{y} f_1(y)=f(1,y), solve_problem2(), which solves min_{x} f_2(x)=f(x,M), and solve_problem3(), which solves min_{x} f_3(x)=f(x,y_{min}(x)). 

**Note 3:** The mathematical expression of the gradient and Hessian of f(x,y), not given in the paper but used in the optimizations, is given in the file notes_optimization.md. The expressions of gradient and Hessian of f_1(y), f_2(x), and f_3(x) can be likewise derived.


[1] E. Peschiera, Y. Agram, F. Quitin, L. Van der Perre, and F. Rottenberg, "On optimizing time-, space- and power-domain energy-saving techniques for sub-6 GHz base stations,” IEEE Transactions on Communications (accepted), 2026.

