#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/27 10:49
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : RVEAa.py
# @Statement : RVEA embedded with the reference vector regeneration strategy
# @Reference : Cheng R, Jin Y, Olhofer M, et al. A reference vector guided evolutionary algorithm for many-objective optimization[J]. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 773-791.
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import cdist


def cal_obj(pop, nobj=3):
    # DTLZ5
    g = np.sum((pop[:, nobj - 1:] - 0.5) ** 2, axis=1)
    temp = np.tile(g.reshape((g.shape[0], 1)), (1, nobj - 2))
    pop[:, 1: nobj - 1] = (1 + 2 * temp * pop[:, 1: nobj - 1]) / (2 + 2 * temp)
    temp1 = np.concatenate((np.ones((g.shape[0], 1)), np.cos(pop[:, : nobj - 1] * np.pi / 2)), axis=1)
    temp2 = np.concatenate((np.ones((g.shape[0], 1)), np.sin(pop[:, np.arange(nobj - 2, -1, -1)] * np.pi / 2)), axis=1)
    return np.tile((1 + g).reshape(g.shape[0], 1), (1, nobj)) * np.fliplr(np.cumprod(temp1, axis=1)) * temp2


def factorial(n):
    # calculate n!
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def combination(n, m):
    # choose m elements from an n-length set
    if m == 0 or m == n:
        return 1
    elif m > n:
        return 0
    else:
        return factorial(n) // (factorial(m) * factorial(n - m))


def reference_points(npop, nvar):
    # calculate approximately npop uniformly distributed reference points on nvar dimensions
    h1 = 0
    while combination(h1 + nvar, nvar - 1) <= npop:
        h1 += 1
    points = np.array(list(combinations(np.arange(1, h1 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
    points = (np.concatenate((points, np.zeros((points.shape[0], 1)) + h1), axis=1) - np.concatenate((np.zeros((points.shape[0], 1)), points), axis=1)) / h1
    if h1 < nvar:
        h2 = 0
        while combination(h1 + nvar - 1, nvar - 1) + combination(h2 + nvar, nvar - 1) <= npop:
            h2 += 1
        if h2 > 0:
            temp_points = np.array(list(combinations(np.arange(1, h2 + nvar), nvar - 1))) - np.arange(nvar - 1) - 1
            temp_points = (np.concatenate((temp_points, np.zeros((temp_points.shape[0], 1)) + h2), axis=1) - np.concatenate((np.zeros((temp_points.shape[0], 1)), temp_points), axis=1)) / h2
            temp_points = temp_points / 2 + 1 / (2 * nvar)
            points = np.concatenate((points, temp_points), axis=0)
    return points


def selection(pop, npop, nvar):
    # select the mating pool
    ind = np.random.randint(0, pop.shape[0], npop)
    mating_pool = pop[ind]
    if npop % 2 == 1:
        mating_pool = np.concatenate((mating_pool, mating_pool[0].reshape(1, nvar)), axis=0)
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < 1 / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def GAoperation(pop, objs, npop, nvar, nobj, lb, ub, eta_c, eta_m):
    # genetic algorithm (GA) operation
    mating_pool = selection(pop, npop, nvar)
    off = crossover(mating_pool, lb, ub, eta_c)
    off = mutation(off, lb, ub, eta_m)
    off_objs = cal_obj(off, nobj)
    return off, off_objs


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 1
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    return rank


def environmental_selection(pop, objs, V, theta):
    # environmental selection
    rank = nd_sort(objs)
    pop = pop[rank == 1]
    objs = objs[rank == 1]
    (npop, nvar) = pop.shape
    nobj = V.shape[1]

    # Step 1. Objective translation
    t_objs = objs - np.min(objs, axis=0)  # translated objectives

    # Step 2. Calculate the smallest angle between each vector
    cosine = 1 - cdist(V, V, 'cosine')
    np.fill_diagonal(cosine, 0)
    gamma = np.min(np.arccos(cosine), axis=1)

    # Step 3. Population partition
    angle = np.arccos(1 - cdist(t_objs, V, 'cosine'))
    association = np.argmin(angle, axis=1)

    # Step 4. Angle-penalized distance calculation
    points = np.unique(association)
    next_pop = np.zeros((points.shape[0], nvar))
    next_objs = np.zeros((points.shape[0], nobj))
    for i in range(points.shape[0]):
        ind = np.where(association == points[i])[0]
        APD = (1 + nobj * theta * angle[ind, points[i]] / gamma[points[i]]) * np.sqrt(np.sum(t_objs[ind] ** 2, axis=1))
        best = ind[np.argmin(APD)]
        next_pop[i] = pop[best]
        next_objs[i] = objs[best]
    return next_pop, next_objs


def regenerate_refs(objs, V):
    # reference vector regeneration
    objs = objs - np.min(objs, axis=0)
    association = np.argmax(1 - cdist(objs, V, 'cosine'), axis=1)
    invalid = np.setdiff1d(np.arange(V.shape[0]), association)
    V[invalid] = np.random.random((len(invalid), V.shape[1])) * np.max(objs, axis=0)
    return V


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, nobj=3, eta_c=30, eta_m=20, alpha=2, fr=0.1):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 30)
    :param eta_m: perturbance factor distribution index (default = 20)
    :param alpha: the parameter to control the change rate of APD (default = 2)
    :param fr: reference vector adaption parameter (default = 0.1)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives
    V0 = reference_points(npop, nobj)  # original reference vectors
    V = np.concatenate((V0, np.random.random((npop, nobj))), axis=0)  # reference vectors

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 100 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. GA operation
        off, off_objs = GAoperation(pop, objs, npop, nvar, nobj, lb, ub, eta_c, eta_m)

        # Step 2.2. Environmental selection
        pop, objs = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), V, (t / iter) ** alpha)

        # Step 2.3. Reference vector adaption
        if t % (iter * fr) == 0:
            V[: npop] = V0 * (np.max(objs, axis=0) - np.min(objs, axis=0))

        # Step 2.4. Reference vector regeneration
        V[npop:] = regenerate_refs(objs, V[npop:])

    # Step 3. Sort the results
    npop = pop.shape[0]
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ5 obtained by RVEAa')
    plt.savefig('Pareto front RVEAa')
    plt.show()


if __name__ == '__main__':
    main(105, 500, np.array([0] * 12), np.array([1] * 12))
