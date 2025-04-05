import numpy as np
import random
from copy import deepcopy

# Parámetros del Sudoku
N = 9  # Tamaño del Sudoku (9x9)
SUBGRID_SIZE = 3  # Tamaño de las subcuadrículas (3x3)


# Función de aptitud: cuenta errores en filas, columnas y subcuadrículas
def fitness(grid):
    def count_errors(arr):
        return N - len(set(arr))  # Duplicados en una fila, columna o subcuadrícula

    errors = 0
    for i in range(N):
        errors += count_errors(grid[i, :])  # Errores en filas
        errors += count_errors(grid[:, i])  # Errores en columnas

    for i in range(0, N, SUBGRID_SIZE):
        for j in range(0, N, SUBGRID_SIZE):
            errors += count_errors(grid[i:i + SUBGRID_SIZE, j:j + SUBGRID_SIZE].flatten())  # Errores en subcuadrículas

    return -errors  # Queremos minimizar los errores


# Función para imprimir el Sudoku con _ en los espacios vacíos
def print_sudoku(grid):
    for row in grid:
        print(" ".join(str(num) if num != 0 else "_" for num in row))
    print()


# Función para definir el Sudoku (debe ser editado manualmente en el código)
def define_sudoku():
    return np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])


# Obtener las posiciones fijas a partir del tablero inicial
def get_fixed_positions(grid):
    fixed_positions = [(i, j, grid[i, j]) for i in range(N) for j in range(N) if grid[i, j] != 0]
    return fixed_positions


# Generar una solución inicial aleatoria respetando las posiciones fijas
def generate_initial_solution(fixed_positions):
    grid = np.zeros((N, N), dtype=int)
    for (i, j, value) in fixed_positions:
        grid[i, j] = value

    for i in range(N):
        missing_values = list(set(range(1, N + 1)) - set(grid[i, :]))
        np.random.shuffle(missing_values)
        for j in range(N):
            if grid[i, j] == 0:
                grid[i, j] = missing_values.pop()
    return grid


# Cruza: intercambia filas de padres para formar un hijo
def crossover(parent1, parent2):
    crossover_point = random.randint(1, N - 1)
    child = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    return child


# Mutación: intercambia dos valores en una fila aleatoria
def mutate(grid, fixed_positions):
    new_grid = deepcopy(grid)
    row = random.randint(0, N - 1)
    indices = [i for i in range(N) if (row, i) not in fixed_positions]
    if len(indices) > 1:
        i1, i2 = random.sample(indices, 2)
        new_grid[row, i1], new_grid[row, i2] = new_grid[row, i2], new_grid[row, i1]
    return new_grid


# Búsqueda local: intenta mejorar una solución intercambiando valores en la misma fila
def local_search(grid, fixed_positions):
    best_grid = deepcopy(grid)
    best_fitness = fitness(grid)
    for _ in range(10):  # Intentos de mejora
        new_grid = mutate(grid, fixed_positions)
        new_fitness = fitness(new_grid)
        if new_fitness > best_fitness:
            best_grid, best_fitness = new_grid, new_fitness
    return best_grid


# Algoritmo genético principal
def genetic_algorithm(fixed_positions, population_size=100, generations=500):
    population = [generate_initial_solution(fixed_positions) for _ in range(population_size)]

    for generation in range(generations):
        population = sorted(population, key=fitness, reverse=True)
        new_population = population[:10]  # Elitismo: mantener los mejores

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:50], 2)  # Selección
            child = crossover(parent1, parent2)
            child = mutate(child, fixed_positions)
            child = local_search(child, fixed_positions)
            new_population.append(child)

        population = new_population
        best_solution = population[0]
        print(f"Generación {generation}: Mejor fitness = {fitness(best_solution)}")
        if fitness(best_solution) == 0:
            break  # Solución óptima encontrada
    return best_solution


sudoku_grid = define_sudoku()
fixed_positions = get_fixed_positions(sudoku_grid)

print("\nSudoku inicial:")
print_sudoku(sudoku_grid)

solution = genetic_algorithm(fixed_positions)

print("Sudoku resuelto:")
print_sudoku(solution)

