import random

# Padres del ejericio anterior.
padre1 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
padre2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

def cruce_uniforme(padre1, padre2):
    hijo1 = []
    hijo2 = []
    for i in range(len(padre1)):
        if random.random() < 0.5:
            hijo1.append(padre1[i])
            hijo2.append(padre2[i])
        else:
            hijo1.append(padre2[i])
            hijo2.append(padre1[i])
    return hijo1, hijo2

def aptitud(individuo):
    aux = 0
    for i in range(len(individuo)):
        aux += individuo[i] * (i ** 2)
    return aux

def imprimir_cruce(padre1, padre2):
    hijo1, hijo2 = cruce_uniforme(padre1, padre2)
    print("Padre 1: ", padre1)
    print("Padre 2: ", padre2)
    print("Hijo 1 : ", hijo1, "| Aptitud:", aptitud(hijo1))
    print("Hijo 2 : ", hijo2, "| Aptitud:", aptitud(hijo2))

imprimir_cruce(padre1, padre2)