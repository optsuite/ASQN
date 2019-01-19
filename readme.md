## ASQN beta 1.0
Structured quasi-Newton method for solving optimization with orthogonality constraint:
\[\min_X f(X)\;  s.t. \; X^{\top}X = I_p, \]
where X is a n-by-p matrix. 

- The current version can be only used to solve problems from the electronic structure calculation and linear eigenvalue problem. However, it can be easily revised for other problems.


## The electronic structure calculation
The code can cover the following two variants:  
  * Kohn-Sham total energy minimization 
  * Hartree-Fock total energy minimization
  
 Running the codes requires a new version of the KSSOLV package [3]. Due to the copyright issues, KSSOLV is not provided.

## Linear eigenvalue problems
 The problem is:
 \[\min \mathrm{Tr}(X^{\top}(A+B)[X]), s.t., X^{\top}X =I_p, \]
 where, $A$ and $B$ are two linear operators or matrices, the computatioal cost of $B[X]$ is much higher than that of $A[X]$.
  * LOBPCG in [4] is used as a subroutine. Its specific license should be considered before modifying and/or redistributing them. 

##References

1. [Jiang Hu, Andre Milzarek, Zaiwen Wen, Yaxiang Yuan. Adaptive Quadratically Regularized Newton Method for Riemannian Optimization. SIAM Journal on Matrix Analysis and Applications](https://epubs.siam.org/doi/10.1137/17M1142478)
2. [Jiang Hu, Bo Jiang, Lin Lin, Zaiwen Wen, Yaxiang Yuan. Structured Quasi-Newton Methods for Optimization with Orthogonality Constraints. arXiv preprint](https://arxiv.org/abs/1809.00452)
3. [Chao Yang, Juan C Meza, Byounghak Lee and Linwang, Wang. KSSOLV—a MATLAB toolbox for solving the Kohn-Sham equations. ACM Transactions on Mathematical Software](https://dl.acm.org/citation.cfm?id=1499099)
4. [A. V. Knyazev. Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate Gradient Method, SIAM Journal on Scientific Computing 23 (2001), no. 2, pp. 517-541](http://dx.doi.org/10.1137/S1064827500366124 )

## The Authors
 We hope that the method is useful for your application.  If you have any bug reports or comments, please feel free to email one of the toolbox authors:

 * Jiang Hu, jianghu at pku.edu.cn
 * Zaiwen Wen, wenzw at pku.edu.cn



## Copyright
-------------------------------------------------------------------------
   Copyright (C) 2018, Jiang Hu, Zaiwen Wen

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without  even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>
