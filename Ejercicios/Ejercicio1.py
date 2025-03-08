I1 = [1,0,0,1,1,1,0,0,1,0,1,1,0,0,1]
I2 = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
I3 = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]
I4 = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
I5 = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
I6 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

def aptitud(individuo):
    aux = 0
    for i in range(len(individuo)):
        aux += individuo[i]*(i**2)
    return aux
        
poblacion = [I1,I2,I3,I4,I5,I6]
for individuo in poblacion:
    print(aptitud(individuo))
