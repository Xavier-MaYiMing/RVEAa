### RVEAa: Reference vector-guided evolution algorithm embedded with the reference vector regeneration strategy

##### Reference: Cheng R, Jin Y, Olhofer M, et al. A reference vector guided evolutionary algorithm for many-objective optimization[J]. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 773-791.

##### RVEAa is an improved version of RVEA which can solve many-objective optimization problems (MaOPs) with irregular Pareto front.

| Variables | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| npop      | Population size                                              |
| iter      | Iteration number                                             |
| lb        | Lower bound                                                  |
| ub        | Upper bound                                                  |
| nobj      | The dimension of objective space (default = 3)               |
| eta_c     | Spread factor distribution index (default = 30)              |
| eta_m     | Perturbance factor distribution index (default = 20)         |
| alpha     | The parameter to control the change rate of APD (default = 2) |
| fr        | Reference vector adaption parameter (default = 0.1)          |
| nvar      | The dimension of decision space                              |
| pop       | Population                                                   |
| objs      | Objectives                                                   |
| V0        | Original reference vectors                                   |
| V         | Reference vectors                                            |
| gamma     | The smallest angle value of each reference vector to the others |
| APD       | Angle-penalized distance                                     |
| dom       | Domination matrix                                            |
| pf        | Pareto front                                                 |

#### Test problem: DTLZ5

$$
\begin{aligned}
	& \theta_i = \frac{\pi}{4(1 + g(x_M))}(1 + 2g(x_M)x_i), \quad i = 1, \cdots, n \\
	& g(x_M) = \sum_{x_i \in x_M} (x_i - 0.5) ^ 2 \\
	& \min \\
	& f_1(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \cos(\theta_{M-2} \pi /2) \cos(\theta_{M - 1} \pi /2) \\
	& f_2(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \cos(\theta_{M-2} \pi /2) \sin(\theta_{M - 1} \pi /2) \\
	& f_3(x) = (1 + g(x_M)) \cos(\theta_i \pi / 2) \cdots \sin(\theta_{M-2} \pi /2) \\
	& \vdots \\
	& f_M(x) = (1 + g(x_M)) \sin(\theta_1 \pi /2) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(105, 500, np.array([0] * 12), np.array([1] * 12))
```

##### Output:

It can be seen that RVEAa performs betters on MaOPs with irregular Pareto fronts than the original RVEA.

![](https://github.com/Xavier-MaYiMing/RVEAa/blob/main/Pareto%20front%20RVEA.png)

![](https://github.com/Xavier-MaYiMing/RVEAa/blob/main/Pareto%20front%20RVEAa.png)



