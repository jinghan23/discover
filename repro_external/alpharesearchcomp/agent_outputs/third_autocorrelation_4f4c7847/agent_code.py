"""AlphaResearch public agent code output: Third Autocorrelation 4f4c7847.

Source: Appendix E of the AlphaResearch OpenReview PDF, section
"Problem : Third Autocorrelation Inequality (4f4c7847)".

This file preserves the published implementation. In particular,
`normalize_population` is intentionally undefined, matching the paper's failure
analysis.
"""

import numpy as np

try:
    from numpy.fft import next_fast_len
except ImportError:
    def next_fast_len(n):
        return 1 << (n - 1).bit_length()


def calculate_c3_upper_bound(height_sequence):
    N = len(height_sequence)
    delta_x = 1 / (2 * N)
    if not hasattr(calculate_c3_upper_bound, "_pad_cache"):
        calculate_c3_upper_bound._pad_cache = {}
    pad_cache = calculate_c3_upper_bound._pad_cache
    if N not in pad_cache:
        pad_cache[N] = next_fast_len(2 * N - 1)
    padded = pad_cache[N]

    integral_f = np.sum(height_sequence) * delta_x
    if integral_f < 1e-12:
        return 0.0
    integral_sq = integral_f * integral_f

    H = np.fft.rfft(height_sequence, n=padded)
    conv_vals = np.fft.irfft(H * H, n=padded)[:2 * N - 1] * delta_x
    max_conv_val = np.max(conv_vals)
    return max_conv_val / integral_sq


def genetic_algorithm(population_size, num_intervals, generations, mutation_rate, crossover_rate):
    population = np.random.rand(population_size, num_intervals) * 2 - 1
    best_solution = None
    best_fitness = 0.0

    for gen in range(generations):
        height_pop = normalize_population(population, 2 * num_intervals)
        fitness_scores = np.array([calculate_c3_upper_bound(h) for h in height_pop])
        current_best_idx = np.argmax(fitness_scores)
        if fitness_scores[current_best_idx] > best_fitness:
            best_fitness = fitness_scores[current_best_idx]
            best_solution = population[current_best_idx].copy()

        indices = np.random.randint(0, population_size, size=(population_size, 2))
        comp_scores = fitness_scores[indices]
        winners = indices[np.arange(population_size), np.argmax(comp_scores, axis=1)]
        new_population = population[winners].copy()

        for i in range(0, population_size, 2):
            if np.random.rand() < crossover_rate:
                parent1 = new_population[i]
                parent2 = new_population[i + 1]
                crossover_point = np.random.randint(1, num_intervals - 1)
                new_population[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                new_population[i + 1] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(num_intervals)
                new_population[i, mutation_point] += np.random.normal(0, 0.1)
                new_population[i, mutation_point] = np.clip(new_population[i, mutation_point], -2, 2)

        population = new_population

    height_best = normalize_population(best_solution[np.newaxis, :], 2 * num_intervals)[0]
    return height_best


def find_better_c3_upper_bound():
    NUM_INTERVALS = 8
    POPULATION_SIZE = 100
    GENERATIONS = 200
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.9

    try:
        from scipy.optimize import differential_evolution

        bounds = [(-2, 2)] * NUM_INTERVALS
        result = differential_evolution(
            lambda x: -calculate_c3_upper_bound(
                normalize_population(x[np.newaxis, :], 2 * NUM_INTERVALS)[0]
            ),
            bounds,
            maxiter=GENERATIONS,
            popsize=max(1, POPULATION_SIZE // 10),
            tol=1e-6,
        )
        height_sequence_3 = normalize_population(result.x[np.newaxis, :], 2 * NUM_INTERVALS)[0]
    except ImportError:
        height_sequence_3 = genetic_algorithm(
            POPULATION_SIZE,
            NUM_INTERVALS,
            GENERATIONS,
            MUTATION_RATE,
            CROSSOVER_RATE,
        )

    return height_sequence_3
