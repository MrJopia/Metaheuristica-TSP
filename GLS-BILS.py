import random
import numpy as np

# Función para leer la matriz de distancias desde un archivo .txt
def read_distance_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convertir cada línea en una lista de enteros
    matrix = [list(map(int, line.split())) for line in lines]
    
    # Convertir la lista a un array de numpy
    distance_matrix = np.array(matrix)
    
    return distance_matrix

# Calcula la longitud del recorrido
def tour_length(weight: np.ndarray, solution: list):
    length = weight[solution[-1], 0] + weight[0, solution[0]]

    for i in range(len(solution) - 1):
        length += weight[solution[i], solution[i + 1]]

    return length

# Inicializa una solución aleatoria
def init_solution(n_cities: int):
    s = np.arange(1, n_cities)
    random.shuffle(s)
    return list(s)

# Indica si la solución tiene una característica
def has_feature(solution, nextcity):
    return True if solution[0] == nextcity or solution[-1] == nextcity else False

# Guided Local Search (GLS) con mejor vecino
def local_search_best_improvement(weight: np.ndarray, init_s: list, penalty: np.ndarray, eta: float, max_unimproved: int):
    n_cities = weight.shape[0]
    s = init_s
    n_unimproved = 0
    s.append(0)  # Cerrar la ruta

    while n_unimproved < max_unimproved:
        best_delta = 0
        best_move = None

        # Recorremos todas las posibles combinaciones de intercambios
        for a in range(len(s) - 1):
            for c in range(a + 1, len(s) - 1):
                # Calculamos el cambio antes y después del intercambio
                before = weight[s[a], s[a+1]] + weight[s[c], s[c+1]]
                after = weight[s[a], s[c]] + weight[s[a+1], s[c+1]]
                dpen = penalty[s[a+1] - 1] - penalty[s[c] - 1]

                # Evaluamos la mejora
                delta = before - (after + eta * dpen)
                if delta > best_delta:
                    best_delta = delta
                    best_move = (a + 1, c)

        # Si encontramos una mejora, la realizamos
        if best_move is not None:
            a, c = best_move
            s[a:c+1] = reversed(s[a:c+1])
            n_unimproved = 0  # Reseteamos el contador
        else:
            n_unimproved += 1  # Si no hay mejora, incrementamos el contador

    s.remove(0)  # Removemos el depósito para no afectar la evaluación
    return s

# Guided Local Search (GLS) con primera mejora
def guided_local_search(weight: np.ndarray, eta: float, max_iter=10000, max_unimproved=100):
    n_cities = weight.shape[0]

    penalty = np.zeros(n_cities - 1)
    utility = np.zeros(n_cities - 1)

    n_iter = 0
    n_unimproved = 0

    current_s = init_solution(n_cities)
    new_s = local_search_best_improvement(weight, init_s=current_s, penalty=penalty, eta=eta, max_unimproved=max_unimproved)

    while n_iter < max_iter or n_unimproved < max_unimproved:

        # Actualiza la mejor solución
        if tour_length(weight, new_s) < tour_length(weight, current_s):
            current_s = new_s
            n_unimproved = 0
        else:
            n_unimproved += 1

        # Calcula la utilidad para cada característica
        for i in range(len(utility)):
            if has_feature(current_s, i + 1):
                utility[i] = weight[0, i + 1] / (1 + penalty[i])

        # Actualiza las penalizaciones
        penalty = np.where(penalty == max(penalty), penalty + 1, penalty)

        # Ejecuta otra iteración de búsqueda local
        new_s = local_search_best_improvement(weight, init_s=current_s, penalty=penalty, eta=eta, max_unimproved=max_unimproved)

        n_iter += 1

    return current_s

# Ejecución de GLS para distintos archivos
def run_gls(paths, eta, max_iter, max_unimproved):
    for path in paths:
        # Leer la matriz de distancias desde el archivo
        weight = read_distance_matrix(path)

        best_solution = None
        best_length = float('inf')

        # Realiza 30 ejecuciones para encontrar la mejor ruta
        for i in range(30):
            opt = guided_local_search(weight, eta, max_iter, max_unimproved)
            current_length = tour_length(weight, opt)

            # Si la nueva solución es mejor, actualizar la mejor
            if current_length < best_length:
                best_solution = opt
                best_length = current_length

        # Mostrar el nombre del archivo y la mejor ruta
        print(f"Archivo: {path}")
        print(f"Mejor ruta: {best_solution}")
        print(f"Longitud de la mejor ruta: {best_length}\n")

# Configuración
eta = 1000
max_iter = 1000
max_unimproved = 500
paths = ['path_to_matrix1.txt', 'path_to_matrix2.txt', 'path_to_matrix3.txt']  # Actualiza con las rutas de tus archivos

# Ejecutar GLS
run_gls(paths, eta, max_iter, max_unimproved)

'''
# Ejemplo de uso:
eta = 1000
max_iter = 1000
max_unimproved = 500

# Suponiendo que tienes una función para leer datos y generar la matriz de pesos
# coord, name = read_data(path)
# w = weight_matrix(coord)

# Solución final con GLS usando búsqueda local con primera mejora
# mejor_ruta = guided_local_search_first_improvement(w, eta, max_iter, max_unimproved)
# print("Mejor ruta encontrada:", mejor_ruta)
'''