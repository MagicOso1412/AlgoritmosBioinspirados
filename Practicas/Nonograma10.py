import random
import numpy as np
import matplotlib.pyplot as plt
import time

def generate_individual(rows, cols):
    """
    Genera un individuo aleatorio para el nonograma.

    Parámetros:
    rows: número de filas del nonograma
    cols: número de columnas del nonograma

    Retorna:
    Una lista plana que representa una matriz binaria para el nonograma
    """
    individual = [random.randint(0, 1) for _ in range(rows * cols)]
    return individual

def calculate_fitness(individual, row_constraints, col_constraints, rows, cols):
    """
    Evalúa la aptitud de una solución candidata para un nonograma.

    Parámetros:
    individual: lista plana representando el estado del nonograma (1=coloreado, 0=vacío)
    row_constraints: lista de restricciones para cada fila
    col_constraints: lista de restricciones para cada columna
    rows: número de filas del nonograma
    cols: número de columnas del nonograma

    Retorna:
    Valor de aptitud (menor es mejor, 0 significa solución perfecta)
    """
    # Convertir la lista plana a una matriz 2D
    grid = np.array(individual).reshape(rows, cols)

    # Calcular f1: diferencia entre las restricciones de fila y los bloques actuales
    f1 = 0
    for i, constraints in enumerate(row_constraints):
        # Obtener los bloques actuales en la fila
        row = grid[i]
        current_blocks = get_blocks(row)

        # Calcular la diferencia
        f1 += calculate_blocks_difference(current_blocks, constraints)

    # Calcular f2: diferencia entre las restricciones de columna y los bloques actuales
    f2 = 0
    for j, constraints in enumerate(col_constraints):
        # Obtener los bloques actuales en la columna
        col = grid[:, j]
        current_blocks = get_blocks(col)

        # Calcular la diferencia
        f2 += calculate_blocks_difference(current_blocks, constraints)

    # La aptitud es la suma de las diferencias (menor es mejor)
    return -(f1 + f2)  # Negativo porque el GA maximiza el fitness

def get_blocks(line):
    """
    Obtiene los bloques consecutivos de 1s en una línea.

    Parámetros:
    line: array numpy de 0s y 1s

    Retorna:
    Lista con las longitudes de los bloques consecutivos de 1s
    """
    blocks = []
    count = 0

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
    """
    Calcula la diferencia entre los bloques actuales y los esperados.

    Parámetros:
    actual_blocks: lista de longitudes de bloques actuales
    expected_blocks: lista de longitudes de bloques esperados

    Retorna:
    Valor que representa la diferencia (menor es mejor)
    """
    # Si no hay bloques pero debería haberlos
    if not actual_blocks and expected_blocks:
        return sum(expected_blocks)
    # Si hay bloques pero no debería haberlos
    elif actual_blocks and not expected_blocks:
        return sum(actual_blocks)
    # Si el número de bloques es incorrecto
    elif len(actual_blocks) != len(expected_blocks):
        return abs(len(actual_blocks) - len(expected_blocks)) * 5  # Penalización mayor
    # Comparar tamaño de los bloques
    else:
        return sum(abs(a - e) for a, e in zip(actual_blocks, expected_blocks))

# Cruza de un punto adaptada para nonogramas
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutación uniforme adaptada para nonogramas
def mutate(individual, mutation_rate):
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_rate:
            mutated_individual.append(1 - gene)  # Invertir el bit
        else:
            mutated_individual.append(gene)
    return mutated_individual

# Selección por torneo adaptada para nonogramas
def tournament_selection(population, fitness_values):
    tournament_indices = random.sample(range(len(population)), 2)  # Torneo binario
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_index]

def print_individual_as_grid(individual, rows, cols):
    """
    Imprime un individuo como una matriz para visualización en consola.
    """
    grid = np.array(individual).reshape(rows, cols)
    for row in grid:
        print(" ".join(["■" if cell == 1 else "□" for cell in row]))
    print()

def genetic_algorithm(population_size, rows, cols, row_constraints, col_constraints,
                     max_generations, mutation_rate, crossover_rate, epsilon=0.0):
    random.seed(123)

    # Inicializar población
    print("=== Inicializando población ===")
    population = []
    for i in range(population_size):
        individual = generate_individual(rows, cols)
        population.append(individual)
        if i < 3:  # Mostrar solo los primeros 3 individuos para no saturar la salida
            print(f"Individuo {i+1}: {individual}")
    print(f"... y {population_size - 3} individuos más")

    fitness_history = []  # Para almacenar el historial de aptitud
    best_solution = None
    best_fitness = float('-inf')

    # Variables para el criterio de detención
    stagnation_counter = 0
    max_stagnation = 50  # Detener si no hay mejora después de 50 generaciones
    generation = 0

    print("\n=== Iniciando evolución ===")
    start_time = time.time()

    termination_reason = "Máximo de generaciones alcanzado"

    while generation < max_generations:
        generation += 1

        # Calcular aptitud para cada individuo
        fitness_values = [calculate_fitness(ind, row_constraints, col_constraints, rows, cols)
                         for ind in population]

        # Encontrar el mejor individuo de esta generación
        current_best_index = fitness_values.index(max(fitness_values))
        current_best = population[current_best_index]
        current_best_fitness = fitness_values[current_best_index]

        # Encontrar el peor individuo de esta generación
        current_worst_index = fitness_values.index(min(fitness_values))
        current_worst_fitness = fitness_values[current_worst_index]

        # Actualizar el mejor global si es necesario
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best.copy()
            stagnation_counter = 0  # Reiniciar contador de estancamiento
        else:
            stagnation_counter += 1

        # Almacenar para el gráfico de convergencia
        fitness_history.append(-best_fitness)  # Convertir de nuevo a positivo para la gráfica

        # Mostrar información de la generación actual
        if generation % 10 == 0 or current_best_fitness == 0:
            print(f"\n=== Generación {generation} ===")
            print(f"Mejor aptitud: {-current_best_fitness} (error total)")
            print(f"Diferencia mejor-peor: {abs(current_best_fitness - current_worst_fitness)}")
            print("Representación en cuadrícula:")
            print_individual_as_grid(current_best, rows, cols)

        # CRITERIO DE PARADA 1: Solución perfecta encontrada
        if current_best_fitness == 0:
            termination_reason = "¡SOLUCIÓN PERFECTA ENCONTRADA!"
            break

        # CRITERIO DE PARADA 2: Similitud entre el mejor y el peor individuo
        if abs(current_best_fitness - current_worst_fitness) <= epsilon:
            termination_reason = f"Convergencia detectada (diferencia ≤ {epsilon})"
            break

        # Verificar criterio de terminación: estancamiento
        if stagnation_counter >= max_stagnation:
            termination_reason = f"Algoritmo detenido después de {max_stagnation} generaciones sin mejora."
            break

        # Crear nueva generación
        new_population = []

        # Elitismo: conservar al mejor individuo
        new_population.append(current_best)

        # Generar descendencia
        while len(new_population) < population_size:
            # Seleccionar padres
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            # Aplicar cruce con cierta probabilidad
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Aplicar mutación
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            # Añadir a la nueva población
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        # Reemplazar la población antigua con la nueva
        population = new_population

    elapsed_time = time.time() - start_time

    # Mostrar el mejor resultado final
    print("\n=== Resultado Final ===")
    print(f"Criterio de parada: {termination_reason}")
    print(f"Mejor aptitud alcanzada: {-best_fitness} (error total)")
    print(f"Generaciones necesarias: {generation} de {max_generations} máximas")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print("Mejor solución encontrada:")
    print_individual_as_grid(best_solution, rows, cols)

    # Retornar el mejor individuo encontrado
    return best_solution, fitness_history, generation, termination_reason

def plot_convergence(fitness_history, generations_executed, termination_reason):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel("Generación")
    plt.ylabel("Error (menor es mejor)")
    plt.title(f"Convergencia del Algoritmo Genético para Nonograma\n({generations_executed} generaciones, {termination_reason})")
    plt.grid(True)
    plt.show()

def plot_solution(solution, rows, cols, row_constraints, col_constraints, title="Solución del Nonograma"):
    grid = np.array(solution).reshape(rows, cols)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Calcular el máximo número de restricciones para espacio
    max_row_hints = max(len(r) for r in row_constraints)
    max_col_hints = max(len(c) for c in col_constraints)

    # Distancia para pistas
    left_margin = max_row_hints * 0.5
    top_margin = max_col_hints * 0.5

    # Dibujar la cuadrícula del nonograma
    ax.imshow(grid, cmap='binary', extent=[0, cols, rows, 0])

    # Dibujar líneas de la cuadrícula
    for i in range(rows + 1):
        ax.axhline(i, color='black', linewidth=1)
    for j in range(cols + 1):
        ax.axvline(j, color='black', linewidth=1)

    # Agregar pistas de fila
    for i, hints in enumerate(row_constraints):
        hint_text = ' '.join(str(h) for h in hints)
        ax.text(-0.2 - 0.5 * len(hints), i + 0.5, hint_text,
                ha='right', va='center', fontsize=10)

    # Agregar pistas de columna
    for j, hints in enumerate(col_constraints):
        hint_text = '\n'.join(str(h) for h in hints)
        ax.text(j + 0.5, -0.2 - 0.4 * len(hints), hint_text,
                ha='center', va='bottom', fontsize=10)

    # Configurar ejes
    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-left_margin, cols)
    ax.set_ylim(rows, -top_margin)

    plt.title(title)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Definir un nonograma (heart)
    rows = 10
    cols = 10
    row_constraints = [[10], [3,3], [2,1,2], [1,2,1,1], [1,2,1], [1,2,1], [1,2,1,1], [2,1,2], [3,3], [10]]
    col_constraints = [[10], [3,3], [2,1,1,2], [1,1,1,1], [1,1], [1,1,1,1], [1,4,1], [2,2,2], [3,3], [10]]

    # Parámetro epsilon para el criterio de parada por convergencia
    epsilon = 0.5  # Diferencia máxima aceptable entre el mejor y el peor fitness

    # Ejecutar el algoritmo genético
    best_solution, fitness_history, generations_executed, termination_reason = genetic_algorithm(
        population_size=1000,  # Población más grande para mejor convergencia
        rows=rows,
        cols=cols,
        row_constraints=row_constraints,
        col_constraints=col_constraints,
        max_generations=200,  # Máximo número de generaciones
        mutation_rate=0.5,
        crossover_rate=0.8,
        epsilon=epsilon  # Parámetro para el criterio de parada por convergencia
    )

    # Mostrar resultados
    print(f"\nEvaluación final de la solución: {fitness_history[-1]}")
    plot_solution(best_solution, rows, cols, row_constraints, col_constraints)
    plot_convergence(fitness_history, generations_executed, termination_reason)