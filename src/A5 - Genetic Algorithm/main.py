"""Implementing Genetic Algorithm

Author: Utkarsh Patel
This module is a part of MIES Genetic Algorithm Coding Assignment
"""

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

def AND(x: str, y: str) -> str:
    """Routine to compute bitwise AND of two bit strings
    ----------------------------------------------------
    Input:
    :param x: first bit string
    :param y: second bit string
    """
    return ''.join(map(min, x, y))

def OR(x: str, y: str) -> str:
    """Routine to compute bitwise OR of two bit strings
    ---------------------------------------------------
    Input:
    :param x: first bit string
    :param y: second bit string
    """
    return ''.join(map(max, x, y))

def NOT(x: str) -> str:
    """Routine to compute bitwise NOT of a bit string
    -------------------------------------------------
    Input:
    :param x: bit string
    """
    return x.replace('1', '2').replace('0', '1').replace('2', '0')

def XOR(x: str, y: str) -> str:
    """Routine to compute bitwise XOR of two bit strings
    ----------------------------------------------------
    Input:
    :param x: first bit string
    :param y: second bit string
    """
    return AND(OR(x, y), OR(NOT(x), NOT(y)))

def calc_fitness_score(s: str) -> int:
    """Routine to compute fitness of a bit string
    ---------------------------------------------
    Input:
    :param s: bit string
    """
    return s.count('1')

def generate_chromosome() -> str:
    """Routine to generate a random bit string"""
    return ''.join(np.random.randint(2, size=args.bit_length).astype(str).tolist())

def crossover(x: str, y: str) -> list[str]:
    """Routine to compute next generation via uniform and one-point crossover
    -------------------------------------------------------------------------
    :param x: parent bit string
    :param y: parent bit string
    """
    children = []

    # uniform crossover
    mask = generate_chromosome()
    children.append(OR(AND(x, mask), AND(y, NOT(mask))))
    children.append(OR(AND(y, mask), AND(x, NOT(mask))))

    # one-point crossover
    pos = np.random.randint(args.bit_length)
    children.append(x[:pos] + y[pos:])
    children.append(y[:pos] + x[pos:])

    return children

def mutate_chromosome(x: str) -> str:
    """Routine to mutate a bit string
    ---------------------------------
    Input:
    :param x: bit string (before mutation)
    """
    mask = np.random.random(size=args.bit_length)
    mask = (mask < args.mutation_rate).astype(int)
    mask = ''.join(mask.astype(str).tolist())
    return XOR(x, mask)

def plot(avg_f: list,
         max_f: list):
    """Routine to plot the average and maximum for each generation"""
    if os.path.exists('plot.png'):
        os.remove('plot.png')
    plt.switch_backend('Agg')
    plt.figure(dpi=100)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.plot(avg_f, label='average')
    plt.plot(max_f, label='maximum')
    plt.title('Fitness vs. generation')
    plt.legend()
    plt.savefig('plot.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--population_size', type=int, default=200, help='population size')
    parser.add_argument('--bit_length', type=int, default=20, help='length of bit string')
    parser.add_argument('--max_iter', type=int, default=300, help='maximum iteration to run')
    parser.add_argument('--mutation_rate', type=float, default=0.01, help='mutation rate')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random number generation')
    args = parser.parse_args()

    np.random.seed(args.random_state)

    # initial generation
    avg_fitness, max_fitness, fittest = [], [], []
    gen = []
    global_max_score, global_max = 0, None
    total_score, max_score, max_id = 0, 0, -1
    for i in range(args.population_size):
        x = generate_chromosome()
        score = calc_fitness_score(x)
        total_score += score
        if score > max_score:
            max_score = score
            max_id = i
        if score > global_max_score:
            global_max_score = score
            global_max = x
        gen.append((score, x))
    avg_fitness.append(total_score / args.population_size)
    max_fitness.append(max_score)
    fittest.append(gen[max_id][1])
    gen = sorted(gen, reverse=True)[: args.population_size // 2]
    np.random.shuffle(gen)

    # looping over generations
    for it in tqdm(range(args.max_iter), desc='Generation'):
        assert len(gen) == args.population_size // 2
        new_gen = []
        total_score, max_score, max_id = 0, 0, -1
        cnt = 0
        for i in range(0, args.population_size // 2, 2):
            children = crossover(gen[i][1], gen[i + 1][1])
            for x in children:
                x = mutate_chromosome(x)
                score = calc_fitness_score(x)
                total_score += score
                if score > max_score:
                    max_score = score
                    max_id = cnt
                if score > global_max_score:
                    global_max_score = score
                    global_max = x
                new_gen.append((score, x))
                cnt += 1
        assert cnt == args.population_size
        gen = new_gen
        avg_fitness.append(total_score / args.population_size)
        max_fitness.append(max_score)
        fittest.append(gen[max_id][1])
        gen = sorted(gen, reverse=True)[: args.population_size // 2]
        np.random.shuffle(gen)

    plot(avg_fitness, max_fitness)
    print('+---------------+----------------------+')
    print('| Generation\t| Fittest chromosome   |')
    print('+---------------+----------------------+')
    for i in range(args.max_iter + 1):
        print(f'| {i:3}\t\t| {fittest[i]} |')
    print('+---------------+----------------------+')
    print(f'\nFinal fittest chromosome: {global_max}\n\nDONE.')

