import random

def generar_tablero(n):
    """Genera un tablero n x n con celdas negras (1) o blancas (0) aleatoriamente."""
    return [[random.choice([0, 1]) for _ in range(n)] for _ in range(n)]

def calcular_pistas(tablero):
    """Calcula las pistas para cada fila o columna."""
    def calcular_linea(linea):
        pistas = []
        contador = 0
        for celda in linea:
            if celda == 1:
                contador += 1
            elif contador > 0:
                pistas.append(contador)
                contador = 0
        if contador > 0:
            pistas.append(contador)
        return pistas if pistas else [0]

    filas = [calcular_linea(fila) for fila in tablero]
    columnas = [calcular_linea([tablero[f][c] for f in range(len(tablero))]) for c in range(len(tablero))]

    return filas, columnas

def mostrar_nonograma(tablero, pistas_filas, pistas_columnas):
    """Muestra el nonograma completo: tablero y pistas."""
    n = len(tablero)

    # Calcular el ancho máximo de las pistas de columna para el espaciado
    max_ancho_col = max(len(str(pista)) for columna in pistas_columnas for pista in columna)

    # Imprimir pistas de columnas (en varias filas si es necesario)
    max_altura_pistas = max(len(col) for col in pistas_columnas)
    for i in range(max_altura_pistas):
        linea = " " * (max_ancho_col + 2)
        for col_pista in pistas_columnas:
            if len(col_pista) < max_altura_pistas - i:
                linea += " " * 3
            else:
                linea += f"{col_pista[i - max_altura_pistas + len(col_pista)]:2} "
        print(linea)

    # Imprimir filas con pistas laterales
    for i, fila in enumerate(tablero):
        pistas = " ".join(map(str, pistas_filas[i])).rjust(max_ancho_col + 2)
        fila_str = "".join("██" if celda == 1 else "  " for celda in fila)
        print(f"{pistas} {fila_str}")

def generar_nonograma(n=5):
    """Genera y muestra un nonograma de tamaño n x n."""
    tablero = generar_tablero(n)
    pistas_filas, pistas_columnas = calcular_pistas(tablero)
    mostrar_nonograma(tablero, pistas_filas, pistas_columnas)

# Ejemplo: Nonograma de 8x8
generar_nonograma(8)
