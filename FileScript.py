import numpy as np

f = open('data chikun.csv', 'r')    #Cargamos los datos

np.set_printoptions(suppress=True)

data = []
for line in f.readlines():
    line = line.strip()
    line = line.split(";")
    line = [float(i) for i in line]
    data.append(line)

x = np.matrix(data)


x = x.reshape(22,6)             #Los acomodamos de la forma que deseamos

np.savetxt('Chikungunya_6_Lag.csv',x,fmt='%.2i',delimiter=';')       #Los guardamos en un csv
