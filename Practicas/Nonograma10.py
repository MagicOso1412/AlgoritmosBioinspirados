import random
import numpy as np
import matplotlib.pyplot as plt


def generate_individual(rows, cols):
    return [random.randint(0, 1) for _ in range(rows * cols)]


def calculate_fitness(individual, row_constraints, col_constraints, rows, cols):
    grid = np.array(individual).reshape(rows, cols)
    f1 = sum(calculate_blocks_difference(get_blocks(grid[i]), row_constraints[i]) for i in range(rows))
    f2 = sum(calculate_blocks_difference(get_blocks(grid[:, j]), col_constraints[j]) for j in range(cols))
    return f1 + f2


def get_blocks(line):
    blocks, count = [], 0
    for cell in line:
        if cell == 1:
            count += 1
        elif count > 0:
            blocks.append(count)
            count = 0
    if count > 0:
        blocks.append(count)
    return blocks


def calculate_blocks_difference(actual_blocks, expected_blocks):
    if not actual_blocks and expected_blocks:
        return sum(expected_blocks) + len(expected_blocks) * 3
    if actual_blocks and not expected_blocks:
        return sum(actual_blocks) + len(actual_blocks) * 3
    if len(actual_blocks) != len(expected_blocks):
        return abs(len(actual_blocks) - len(expected_blocks)) * 15
    return sum(abs(a - e) for a, e in zip(actual_blocks, expected_blocks))


def crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


def tournament_selection(population, fitness_values, tournament_size=4):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    best_index = min(tournament_indices, key=lambda i: fitness_values[i])
    return population[best_index]


def genetic_algorithm(population_size, rows, cols, row_constraints, col_constraints, mutation_rate, crossover_rate,
                      max_generations=1500):
    # seed = 123
    seed = random.randint(0, 1000000)  # Genera una semilla aleatoria
    random.seed(seed)  # Establece la semilla
    print(f"Semilla utilizada: {seed}")  # Imprime la semilla generada
    population = [generate_individual(rows, cols) for _ in range(population_size)]
    best_solution, best_fitness, generation = None, float('inf'), 0
    fitness_history = []

    for generation in range(max_generations):
        fitness_values = [calculate_fitness(ind, row_constraints, col_constraints, rows, cols) for ind in population]
        best_index = np.argmin(fitness_values)
        best_solution, best_fitness = population[best_index], fitness_values[best_index]
        fitness_history.append(best_fitness)

        if best_fitness == 0:
            break

        new_population = [best_solution]  # Elitismo
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            new_population.append(mutate(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child2, mutation_rate))
        population = new_population

    print(f"Mejor aptitud alcanzada: {best_fitness}")
    print(f"Número de generaciones: {generation}")
    return best_solution, fitness_history, generation


def plot_solution(solution, rows, cols):
    grid = np.array(solution).reshape(rows, cols)
    grid = np.transpose(grid)  # Corrige la orientación
    plt.imshow(grid, cmap='binary', origin='upper')  # Asegura que la imagen se dibuje correctamente
    plt.grid(True)
    plt.title("Solución del Nonograma")
    plt.show()



# Parámetros y ejecución
if __name__ == "__main__":
    rows, cols = 10, 10
    row_constraints = [[10], [3, 3], [2, 1, 2], [1, 2, 1, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1, 1], [2, 1, 2], [3, 3],
                       [10]]
    col_constraints = [[10], [3, 3], [2, 1, 1, 2], [1, 1, 1, 1], [1, 1], [1, 1, 1, 1], [1, 4, 1], [2, 2, 2], [3, 3],
                       [10]]

    best_solution, fitness_history, generations_executed = genetic_algorithm(
        population_size=1200,
        rows=rows,
        cols=cols,
        row_constraints=row_constraints,
        col_constraints=col_constraints,
        mutation_rate=0.015,
        crossover_rate=0.95,
        max_generations=1500
    )

    plot_solution(best_solution, rows, cols)
    plt.plot(fitness_history)
    plt.xlabel("Generación")
    plt.ylabel("Error")
    plt.title("Convergencia del Algoritmo Genético")
    plt.grid(True)
    plt.show()
