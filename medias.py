import pandas as pd
import statistics as st
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


#Calcula un centro de Masa para un conjunto de datos dado.
def centroMasa(datos):
	count = 0
	Xs = []
	Ys = []
	Zs = []
	lista = datos.tolist()
	for i in range(10):
		Xs.append(lista[count])
		Ys.append(lista[count+1])
		Zs.append(lista[count+2])
		count = count+3

	CM = [st.mean(Xs), st.mean(Ys), st.mean(Zs)]
	return CM

#Traslada a un conjunto de puntos.
def translacion(centroMasa, datos):
	count2 = 0
	coordenadas = []
	lista2 = datos.tolist()
	for i in range(10):
		coordenadas.append(lista2[count2]-centroMasa[0])
		coordenadas.append(lista2[count2+1]-centroMasa[1])
		coordenadas.append(lista2[count2+2]-centroMasa[2])
		count2 = count2+3
	return coordenadas

#Calcula la distancia euclídea entre dos puntos.
def euclidea(cx ,cy, cz, fx, fy, fz):
	dist = math.sqrt(pow((fx-cx),2) + pow((fy-cy),2) + pow((fz-cz),2))
	return dist


#Leemos los datos de los landmarks del excel.
alldata = pd.read_excel ("RAW_3D_Full_data_PG2011.xlsx")

#Filtramos para quedarnos solo con las columnas que necesitamos.
landmarks = alldata[['xCg2', 'yCg2', 'zCg2',
             'xCzyG2', 'yCzyG2', 'zCzyG2',
             'xCzyD2', 'yCzyD2', 'zCzyD2',
             'xYfmtG2', 'yYfmtG2', 'zYfmtG2',
             'xYfmtD2', 'yYfmtD2', 'zYfmtD2',
             'xNn2', 'yNn2', 'zNn2',
             'xBgn2', 'yBgn2', 'zBgn2',
             'xBgoG2', 'yBgoG2', 'zBgoG2',
             'xBgoD2', 'yBgoD2', 'zBgoD2',
             'xBpg2', 'yBpg2', 'zBpg2',
             'xCg', 'yCg', 'zCg',
             'xCzyG', 'yCzyG', 'zCzyG',
             'xCzyD', 'yCzyD', 'zCzyD',
             'xYfmtG', 'yYfmtG', 'zYfmtG',
             'xYfmtD', 'yYfmtD', 'zYfmtD',
             'xNn', 'yNn', 'zNn',
             'xBgn', 'yBgn', 'zBgn',
             'xBgoG', 'yBgoG', 'zBgoG',
             'xBgoD', 'yBgoD', 'zBgoD',
             'xBpg', 'yBpg', 'zBpg']]

#Eliminamos aquellas filas en las que falta algún dato.
NoNAN = landmarks.dropna()
NoNAN = NoNAN[0:84]


#Interamos por todas las filas para calcular el centro de masas de cada muestra.
#CMs2 para los landmarks faciales y CMs1 para los landmarks craneales.
CMs1 = []
CMs2 = []

for i in range (84):
	CMs2.append(centroMasa(NoNAN.iloc[i,0:30]))
	CMs1.append(centroMasa(NoNAN.iloc[i,30:60]))


landmarksC = []
landmarksF = []
for i in range (84):
	landmarksC.append(translacion(CMs1[i], NoNAN.iloc[i,30:60]))
	landmarksF.append(translacion(CMs2[i], NoNAN.iloc[i,0:30]))


#Mostramos los puntos en un sistema de coordenadas 3d.
"""colors=["#0000FF", "#00FF00"]

x = []
y = []
z = []

for j in range(len(landmarksC)):
	contador = 0
	for i in range(10):
		x.append(landmarksC[j][contador])
		y.append(landmarksC[j][contador+1])
		z.append(landmarksC[j][contador+2])
		contador = contador+3

x2 = []
y2 = []
z2 = []

for j in range(len(landmarksF)):
	contador = 0
	for i in range(10):
		x2.append(landmarksF[j][contador])
		y2.append(landmarksF[j][contador+1])
		z2.append(landmarksF[j][contador+2])
		contador = contador+3

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(x, y, z, color=colors[0])
ax.scatter(x2, y2, z2, color=colors[1])
pyplot.show()"""

FSTDs = []
FSTD = []
#Calculamos ahora el FSTD
for i in range(len(landmarksC)):
	contador2 = 0
	for j in range(10):
		FSTD.append(euclidea(landmarksC[i][contador2], landmarksC[i][contador2+1], landmarksC[i][contador2+2], landmarksF[i][contador2], landmarksF[i][contador2+1], landmarksF[i][contador2+2]))
		contador2 = contador2+3
	FSTDs.append(FSTD)
	FSTD = []

medias=[]
#Calculamos la media para cada punto.
for i in range(10):
	suma=0
	for j in range(len(FSTDs)):
		suma += FSTDs[j][i]
	suma/=len(FSTDs)
	medias.append(suma)

print(medias)

#Pinta una gráfica con los FSTD tanto reales como estimadas.
"""mostrar = []
mostrar2 = []
for i in range(10):
	mostrar = []
	mostrar2 = []
	for j in range(len(FSTDs)):
		mostrar.append(FSTDs[j][i])
		mostrar2.append(e_resultados[j][i])

	pyplot.plot(mostrar, 'ro')
	pyplot.plot(mostrar2, 'bo')
	pyplot.show()"""


reales=[[],[],[],[],[],[],[],[],[],[]]
estimado=[[],[],[],[],[],[],[],[],[],[]]

for i in range(10):
	reales[i] += [FSTDs[j][i] for j in range(len(FSTDs))]
	for j in range(len(FSTDs)):
		estimado[i].append(medias[i])

rmse = []
mae = []


for i in range(len(reales)):
	rmse.append(math.sqrt(mean_squared_error(reales[i], estimado[i])))
	mae.append(mean_absolute_error(reales[i], estimado[i]))


mostrar = {'Punto' : ['Cg','CzyG','CzyD','YfmtG','YfmtD','Nn','Bgn','BgoG','BgoD','Bpg'], 'RMSE' : rmse, 'MAE': mae}
print(tabulate(mostrar, headers='keys'))
#print(rmse)
#print(mae)
