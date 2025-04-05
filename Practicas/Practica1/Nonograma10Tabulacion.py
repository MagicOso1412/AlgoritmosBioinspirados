import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tabulate import tabulate


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
                      max_generations=1500, seed=None):
    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randint(0, 1000000)
        random.seed(seed)

    # Inicializar población
    population = [generate_individual(rows, cols) for _ in range(population_size)]
    best_solution, best_fitness = None, float('inf')

    # Historiales para seguimiento de estadísticas
    best_fitness_history = []

    # Variables para estadísticas
    success_generation = None
    evaluations_count = 0  # Contador de evaluaciones de la función objetivo

    for generation in range(max_generations):
        # Calcular fitness de toda la población
        fitness_values = []
        for ind in population:
            fitness = calculate_fitness(ind, row_constraints, col_constraints, rows, cols)
            fitness_values.append(fitness)
            evaluations_count += 1  # Incrementar el contador de evaluaciones

        # Encontrar el mejor
        best_index = np.argmin(fitness_values)
        generation_best_solution = population[best_index]
        generation_best_fitness = fitness_values[best_index]

        # Actualizar la mejor solución global si es necesario
        if generation_best_fitness < best_fitness:
            best_solution = generation_best_solution.copy()
            best_fitness = generation_best_fitness

        # Guardar valor para histórico
        best_fitness_history.append(generation_best_fitness)

        # Verificar si se ha alcanzado el éxito
        if generation_best_fitness == 0 and success_generation is None:
            success_generation = generation

        if generation_best_fitness == 0:
            break

        # Crear nueva población
        new_population = [generation_best_solution]  # Elitismo
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

    # Calcular estadísticas finales
    total_generations = generation + 1
    success = best_fitness == 0

    stats = {
        'best_fitness_history': best_fitness_history,
        'total_generations': total_generations,
        'success': success,
        'best_fitness': best_fitness,
        'success_generation': success_generation,
        'seed': seed,
        'evaluations_count': evaluations_count
    }

    return best_solution, stats


def run_multiple_executions(num_executions, population_size, rows, cols, row_constraints, col_constraints,
                           mutation_rate, crossover_rate, max_generations=1500):
    """Ejecuta el algoritmo genético múltiples veces y calcula estadísticas globales."""
    print(f"Ejecutando {num_executions} ejecuciones independientes del algoritmo genético...")

    # Variables para estadísticas globales
    successful_runs = 0
    fitness_values = []
    evaluations_counts = []
    all_stats = []
    best_solution_overall = None
    best_fitness_overall = float('inf')
    no_diversity_loss_count = 0

    # Generar semillas diferentes para cada ejecución
    seeds = [random.randint(0, 1000000) for _ in range(num_executions)]

    for execution, seed in enumerate(seeds):
        print(f"\nEjecución {execution + 1}/{num_executions} - Semilla: {seed}")

        solution, stats = genetic_algorithm(
            population_size=population_size,
            rows=rows,
            cols=cols,
            row_constraints=row_constraints,
            col_constraints=col_constraints,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_generations=max_generations,
            seed=seed
        )

        all_stats.append(stats)
        fitness_values.append(stats['best_fitness'])
        evaluations_counts.append(stats['evaluations_count'])

        if stats['success']:
            successful_runs += 1

        if stats['best_fitness'] < best_fitness_overall:
            best_fitness_overall = stats['best_fitness']
            best_solution_overall = solution

        print(f"Mejor aptitud: {stats['best_fitness']}")
        print(f"Éxito: {'Sí' if stats['success'] else 'No'}")
        print(f"Evaluaciones: {stats['evaluations_count']}")

    # Calcular estadísticas globales
    success_rate = (successful_runs / num_executions) * 100
    avg_fitness = np.mean(fitness_values)
    min_fitness = np.min(fitness_values)
    max_fitness = np.max(fitness_values)
    std_fitness = np.std(fitness_values)
    avg_evaluations = np.mean(evaluations_counts)

    # Mostrar estadísticas globales
    print("\n" + "="*50)
    print("ESTADÍSTICAS GLOBALES")
    print("="*50)
    print(f"Número total de ejecuciones: {num_executions}")
    print(f"Número de ejecuciones exitosas: {successful_runs}")
    print(f"Porcentaje de éxito: {success_rate:.2f}%")
    print(f"Promedio de evaluaciones de la función objetivo: {avg_evaluations:.2f}")
    print(f"Mínimo de la función objetivo: {min_fitness}")
    print(f"Máximo de la función objetivo: {max_fitness}")
    print(f"Promedio de la función objetivo: {avg_fitness:.2f}")
    print(f"Desviación estándar de la función objetivo: {std_fitness:.2f}")

    global_stats = {
        'success_rate': success_rate,
        'avg_evaluations': avg_evaluations,
        'min_fitness': min_fitness,
        'max_fitness': max_fitness,
        'avg_fitness': avg_fitness,
        'std_fitness': std_fitness,
        'best_solution': best_solution_overall,
        'best_fitness': best_fitness_overall,
        'all_stats': all_stats,
        'num_executions': num_executions
    }

    return global_stats


def plot_solution(solution, rows, cols):
    grid = np.array(solution).reshape(rows, cols)
    grid = np.transpose(grid)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary', origin='upper')
    plt.grid(True)
    plt.title("Solución del Nonograma")
    plt.show()

def plot_algorithm_evolution(global_stats):
    """
    Grafica la evolución del algoritmo genético mostrando el mejor, peor y caso mediano.
    El eje Y representa el valor de la función objetivo del mejor individuo por generación.
    """
    all_stats = global_stats['all_stats']
    num_executions = global_stats['num_executions']

    # Ordenar ejecuciones por fitness final para encontrar mejor, peor y mediana
    sorted_indices = sorted(range(num_executions), key=lambda i: all_stats[i]['best_fitness'])

    # Índices de las ejecuciones mejor, peor y mediana
    best_run_idx = sorted_indices[0]
    worst_run_idx = sorted_indices[-1]
    median_run_idx = sorted_indices[num_executions // 2]

    # Historiales de fitness
    best_run_history = all_stats[best_run_idx]['best_fitness_history']
    worst_run_history = all_stats[worst_run_idx]['best_fitness_history']
    median_run_history = all_stats[median_run_idx]['best_fitness_history']

    # Crear la gráfica
    plt.figure(figsize=(12, 7))

    plt.plot(best_run_history, 'g-', label=f'Mejor ejecución (fitness final: {all_stats[best_run_idx]["best_fitness"]})')
    plt.plot(worst_run_history, 'r-', label=f'Peor ejecución (fitness final: {all_stats[worst_run_idx]["best_fitness"]})')
    plt.plot(median_run_history, 'b-', label=f'Ejecución mediana (fitness final: {all_stats[median_run_idx]["best_fitness"]})')

    plt.xlabel('Generación')
    plt.ylabel('Valor de la función objetivo (menor es mejor)')
    plt.title('Evolución del algoritmo genético')
    plt.legend()
    plt.grid(True)

    # Agregar una línea horizontal en y=0 para referencia del óptimo
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.savefig('evolucion_algoritmo.png')
    plt.show()

    print(f"\nMejor ejecución: {all_stats[best_run_idx]['best_fitness']} (semilla: {all_stats[best_run_idx]['seed']})")
    print(f"Ejecución mediana: {all_stats[median_run_idx]['best_fitness']} (semilla: {all_stats[median_run_idx]['seed']})")
    print(f"Peor ejecución: {all_stats[worst_run_idx]['best_fitness']} (semilla: {all_stats[worst_run_idx]['seed']})")


def create_results_table(global_stats, config_name="1"):
    """Crea una tabla de resultados según el formato solicitado."""
    data = {
        'Conf': [config_name],
        'Porcentaje de éxitos': [f"{global_stats['success_rate']:.1f}%"],
        'Promedio función objetivo': [f"{global_stats['avg_fitness']:.1f}"],
        'Desviación estándar': [f"{global_stats['std_fitness']:.1f}"],
        'Mínimo de la función': [f"{global_stats['min_fitness']}"],
        'Máximo de la función': [f"{global_stats['max_fitness']}"],
        'Promedio evaluaciones': [f"{global_stats['avg_evaluations']:.1f}"]
    }

    df = pd.DataFrame(data)
    table_str = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

    print("\nTabla de resultados:")
    print(table_str)

    return df


def compare_configurations(configs, rows, cols, row_constraints, col_constraints, num_executions=20):
    """Compara diferentes configuraciones del algoritmo genético."""
    results = []
    all_global_stats = []

    for i, config in enumerate(configs):
        print(f"\n{'='*50}")
        print(f"CONFIGURACIÓN {i+1}: {config['name']}")
        print(f"{'='*50}")

        global_stats = run_multiple_executions(
            num_executions=num_executions,
            population_size=config['population_size'],
            rows=rows,
            cols=cols,
            row_constraints=row_constraints,
            col_constraints=col_constraints,
            mutation_rate=config['mutation_rate'],
            crossover_rate=config['crossover_rate'],
            max_generations=config['max_generations']
        )

        # Guardar las estadísticas globales
        all_global_stats.append(global_stats)

        # Generar gráfica de evolución para esta configuración
        print(f"\nGenerando gráfica de evolución para configuración {i+1}...")
        plot_algorithm_evolution(global_stats)

        # Crear tabla para esta configuración
        df = create_results_table(global_stats, config_name=str(i+1))
        results.append(df)

    # Combinar resultados
    combined_results = pd.concat(results)

    # Mostrar tabla comparativa completa
    print("\n\nTABLA COMPARATIVA DE TODAS LAS CONFIGURACIONES:")
    print(tabulate(combined_results, headers='keys', tablefmt='pipe', showindex=False))

    return combined_results, all_global_stats


# Parámetros y ejecución
if __name__ == "__main__":
    rows, cols = 10, 10
    row_constraints = [[10], [3, 3], [2, 1, 2], [1, 2, 1, 1], [1, 2, 1], [1, 2, 1], [1, 2, 1, 1], [2, 1, 2], [3, 3], [10]]
    col_constraints = [[10], [3, 3], [2, 1, 1, 2], [1, 1, 1, 1], [1, 1], [1, 1, 1, 1], [1, 4, 1], [2, 2, 2], [3, 3], [10]]

    # Definir diferentes configuraciones
    configs = [
        {
            'name': 'Config estándar',
            'population_size': 1200,
            'mutation_rate': 0.015,
            'crossover_rate': 0.95,
            'max_generations': 1500
        },

        {
            'name': 'Config alta mutación',
            'population_size': 1200,
            'mutation_rate': 0.03,
            'crossover_rate': 0.95,
            'max_generations': 1500
        },

        {
            'name': 'Config baja mutación',
            'population_size': 1200,
            'mutation_rate': 0.005,
            'crossover_rate': 0.95,
            'max_generations': 1500
        }
    ]

    # Para ejecutar una sola configuración:
    # global_stats = run_multiple_executions(
    #     num_executions=20,  # 20 ejecuciones independientes
    #     population_size=1200,
    #     rows=rows,
    #     cols=cols,
    #     row_constraints=row_constraints,
    #     col_constraints=col_constraints,
    #     mutation_rate=0.015,
    #     crossover_rate=0.95,
    #     max_generations=1500
    # )
    # create_results_table(global_stats)
    # plot_algorithm_evolution(global_stats)  # Generar gráfica para una configuración

    # O para comparar varias configuraciones:
    combined_results, all_global_stats = compare_configurations(configs, rows, cols, row_constraints, col_constraints, num_executions=20)

    # Si quieres visualizar la mejor solución encontrada entre todas las configuraciones:
    best_config_idx = np.argmin([stats['best_fi tness'] for stats in all_global_stats])
    best_solution = all_global_stats[best_config_idx]['best_solution']
    print(f"\nLa mejor solución fue encontrada en la configuración {best_config_idx + 1}")
    plot_solution(best_solution, rows, cols)