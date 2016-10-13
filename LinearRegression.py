import numpy as np
import random
import matplotlib.pyplot as plt
def featureNormalize(X):

    [m,n] = X.shape

    mean = X.mean(0)
    std = np.amax(X, axis= 0) - np.amin(X, axis= 0)

    X = (X-mean)/std

    return [X,mean,std]

def testNormalize(Xt,mean,sigma):
    Xt = (Xt-mean)/sigma
    return (Xt)

def computeCost(X,y,theta):
    [m,n]= y.shape
    J = (1/(2*m))*sum(np.power((X*theta-y),2))
    return J

def gradientDescent (X, y, theta, alpha, num_iters):
    J_histor = np.zeros((num_iters,1))
    [m,n] = X.shape         # m = numero de ejemplos de entrenamiento
    i=0
    theta_temp = []
    j=0
    for i in range (num_iters):
        h = X*theta
        error=h-y
        theta_temp = np.transpose(((alpha/m)*(np.transpose(error)*X)))
        theta = theta-theta_temp
        J_histor [i] = computeCost(X,y,theta)
    return [theta,J_histor]

def Pseudoinverse(X,Y):                            #Función que calcula la pseudoinversa de una matriz
    Xplus=np.linalg.pinv(X)
    th = Xplus.dot(Y)
    return th

def MAPE(y,h):
    [m,n]=y.shape
    error=(y-h)/y
    error=np.abs(error)
    error=100/m*np.sum(error)
    return error

def MSE(y,h):
    [m,n] = y.shape
    error = (1/m) * sum (np.power((h-y),2))
    return error

def main():

    f = open('Dengue_1_Lag.csv', 'r')           #Ingresamos el nombre del dataset a utilizar

    np.set_printoptions(suppress=True)

    data = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(";")
        line = [float(i) for i in line]
        data.append(line)

    x = np.matrix(data)         #Matriz x (entradas)

    [m,n] = x.shape
    t = np.arange(0,round((m*0.6)))

    training_ds = x[t,:]        #Matriz con datos de entrenamiento (60% de los datos iniciales)
    t2 = np.arange((round((m*0.6))),(round((m*0.8))))
    cv_ds = x[t2,:]            #Matriz con datos de CV
    t3 = np.arange(round((m*0.8)),m)
    tests_ds = x[t3,:]       #Matriz con datos de prueba

    y_training=training_ds[:,n-1]
    y_cv = cv_ds[:,n-1]
    y_test= tests_ds[:,n-1]
    x = np.delete(x,n-1,1)       #Matriz x (entradas) (vector y eliminado de x)
    training_ds = np.delete(training_ds,n-1,1)
    cv_ds = np.delete(cv_ds,n-1,1)
    tests_ds = np.delete(tests_ds,n-1,1)
    [training_ds,mean,sigma]=featureNormalize(training_ds) #Normalizar datos del conjunto de entrenamiento
    cv_ds = testNormalize(cv_ds,mean,sigma)
    # Por el momento no se hara uso del conjunto de test
    # #tests_ds = testNormalize(tests_ds,mean,sigma)
    training_ds=np.concatenate((np.ones((round((m*0.6)),1)),training_ds),axis=1)
    cv_ds=np.concatenate((np.ones((round((m*0.2)),1)),cv_ds),axis=1)
    # tests_ds = np.concatenate((np.ones(((m-round((m*0.6))),1)),tests_ds),axis=1)
    thetaG_y1 = np.zeros((n,1))
    alpha = 0.09
    iterat = 400
    [thetaG_y1,J] = gradientDescent(training_ds,y_training,thetaG_y1,alpha,iterat)
    thetaN_y1 = Pseudoinverse(training_ds,y_training)

    print ("Thetas calculados por el metodo de Gradiente Descendiente: \n", thetaG_y1)
    print ("Thetas calculados por el metodo de la Pseudoinversa: \n",thetaN_y1)

    errorG_y1=MSE(y_test,np.round(cv_ds*thetaG_y1))
    errorN_y1=MSE(y_test,np.round(cv_ds*thetaN_y1))

    print("Error MSE para predicciones con Gradiente Descendiente",errorG_y1)
    print("Error MSE para predicciones con Ecuaciones Normales",errorN_y1)

    plt.figure(1)
    plt.ylabel('Cost J')
    plt.xlabel('Number of Iterations')
    plt.title('Cambio de la funcion de costo')
    plt.plot(J)
    plt.show()


main()