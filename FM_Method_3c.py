import numpy as np

def __Next_to( x , shape ):
    #Obtiene los puntos al lado de x
    x = list(x) #Para que este código funcione es necesario que x esté en forma de lista
    Next = []
    for i in range(len(x)):
        if x[i] > 0 :
            nearP = x.copy()
            nearP[i] -= 1
            Next = Next + [nearP]
        if x[i] < shape[i] - 1 :
            nearP = x.copy()
            nearP[i] += 1
            Next = Next + [nearP]
    return np.array(Next)

def __Cuadratic_Solver(costs, Delta2I, f2I):
    #Resuelve la ecuación cuadrática AU**2 + B*U + C = 0 que obtiene el nuevo valor de coste
    A = len(costs) * Delta2I
    B = -2  * sum(costs) * Delta2I
    C = sum(costs**2) * Delta2I - f2I
    return ( -B + np.sqrt( B**2 - 4*A*C ) ) / (2*A)

def __cost( x , f , Delta ):
    #Obtiene el coste del punto x, f valor de 1/F en el punto x
    #Obtiene el coste a partir de una aproximación de la ecuación a los puntos inmediatamente cercanos, se puede mejorar la aproximación añadiendo más puntos
    Near = __Next_to( x , T.shape )
    vector_costs = []
    for i in range(len(x)):
        line = Near[ Near[:,i] != x[i] ]
        min_cost = min( T[ tuple(line.transpose()) ] )
        if min_cost != Max:
            vector_costs.append( min_cost )

    if len(vector_costs) == 0: return T[tuple(x)] #Caso raro en el que f<=0 y no hay puntos evaluados al rededor
    if f <= 0 : return( min(vector_costs) );
    #Las siguientes dos lineas se complicarían bastante si no se considera que todos los puntos son equidistantes a distancia Delta, se podría trabajar en implementar el modelo para redes no homogeneas
    if len( vector_costs ) == 1 : return Delta*f + vector_costs[0];
    return __Cuadratic_Solver( np.array(vector_costs) , 1 / Delta**2 , f**2 )

class __HashTable():
    #La velocidad de este algoritmo se fundamenta en mantener la información bien organizada
    #Se intenta con una hash table, de tal forma que los hijos del punto i, son 2i y 2i+1
    #Se organiza con dos listas Costs, Points y una np array Positions
    #Positions es un np array del misma tamaño que T, guarda la posición de cada punto en los otros vectores de Trials, si no está en Trials, se le asigna -1
    #Los vectores Costs y Points organizan en listas la información guardada, la primera contiene los costes provisionales y la segunda su posición en el array

    def __init__(self,shape):
        #Se tiene que la base del árbol toma la posición 1, se ocupa el nodo 0 de tal forma que quede invariante a lo largo del algoritmo
        self.Costs = [-1]
        self.Points = [None]
        self.Positions = - np.ones( shape , dtype = int)

    def Pop( self ): #Extrae el punto de la base del árbol y actualiza
        #Se guardan los valores a extraer
        cost = self.Costs[1]
        point = self.Points[1]

        #Se descartan los valores del nodo base, se actualiza la tabla y retorna los valores que se quieren sacar
        self.Positions[ point ] = -1
        self.__Change( 1 , "None" , Max )
        self.__UpdateDown( 1 )
        return point , cost

    def Add( self , point , cost ): #Añade un punto a la tabla
        node = len(self.Costs)
        self.Points.append( point )
        self.Costs.append( cost )
        self.Positions[ point ] = node
        self.__UpdateUp(node) #Sube el punto hasta su posición ordenada

    def Update( self , point , cost): #Actualiza el punto
        #Solo se actualiza si tiene un coste menor
        #En teoría, por como está construido el algoritmo, siempre que se llame a esta función, el coste nuevo será menor,
            #por si acaso, se comprueba
        node = self.Positions[point]
        if self.Costs[ node ] > cost:
            self.__Change( node , point , cost )
            self.__UpdateUp( node )

    def Clean( self ): #Limpia el final de la tabla, se podría mejorar esta parte
        while self.Points[-1] == "None":
            del self.Costs[-1]
            del self.Points[-1]

    def __Change( self , node , point , cost ): #Cambia los valores del nodo
        self.Points[ node ] = point
        self.Costs[ node ] = cost
        if point != "None" : #Nodo no descartado
            self.Positions[ point ] = node

    def __UpdateUp( self , son ): #Actualiza la Hash Table hacia arriba
        father = son // 2
        if self.Costs[ father ] > self.Costs[ son ]:
            sonCost = self.Costs[son]
            sonPoint = self.Points[son]
            self.__Change( son , self.Points[father] , self.Costs[father] )
            self.__Change( father , sonPoint , sonCost )
            self.__UpdateUp( father ) #Proceso recursivo, se comprueba si el nuevo nodo padre sigue subiendo por el árbol

    def __UpdateDown( self , father ): #Actualiza la Hash Table hacia abajo
        son = father*2
        #Primero comprueba cuál de los hijos es menor
        if len(self.Costs) > son: #Si no existen nodos hijos, no hace nada
            if len(self.Costs) > son+1 and self.Costs[son] > self.Costs[son+1]:
                son += 1

            #Sube el nodo hijo a la posición del padre
            cost = self.Costs[ son ]
            if cost != Max: #Por como esta construido el algoritmo, el nodo padre es un descarte, casi siempre baja
                self.__Change( father , self.Points[son] , cost )
                self.__Change( son , "None" , Max ) #El hijo pasa a ser el descarte
                self.__UpdateDown( son )



def solve_Eikonal( F , Gamma , Delta = 1. , cutOff = -1. ):
    """
    Metodo númerico para aproximar la solución de la ecuación Eikonal |Delta T|F = 1
    Se toma como entrada
        -F como un np array del tamaño que se quiere resolver con el valor de la función velocidad en cada punto
        -Gamma np.array con los puntos de la condición inicial T=0
        -Delta el valor de la distancia entre puntos, se debe tomar una red de puntos uniforme
        -cutOff limita el número de cálculos que se realizarán, si es -1 se considera que no hay cota
    """
    #Se inicializa el algoritmo
    global Max , T
    F = F.astype(float)
    F[F != 0] = 1 / F[F != 0] #Se obtiene aquí la inversa de F para no tener que calcularlo repetidamente
    Max = Delta * sum( np.ceil(F[F != 0]) ) #Cota superior de T
    if cutOff == -1 : cutOff = Max;
    T = Max * np.ones_like( F )  #Los puntos lejanos quedan marcados con Max, cota superior de que puede tomar este modelo
    T[ tuple( Gamma.transpose() ) ] = 0  #Se iguala la condición inicial a 0
    Trials = __HashTable( T.shape ) # Se define una hash table Trials que mantiene la lista de posibles puntos ordenada

    #Prepara el porcentaje de desarrollo del programa
    N_Max = T.size - len( Gamma )
    progress = 0

    #Se inicializa el algoritmo FM
    #Los puntos están separados en tres subgrupos
        #->Known, ya tienen un valor aasigando, al inicio solo los puntos de gamma son conocidos y tienen el valor 0 asignado
        #->Trials, alguno de sus puntos vecinos es un Known, y tienen asignado un coste provisional
            #El coste provisional de estos puntos es calculado por la funcion __cost, y depende de la función velocidad y de sus vecinos Known
        #->Far, el resto
    for known in Gamma: #Introduce en Trials todos los puntos que rodean a Gamma
        for near in __Next_to(known, T.shape):
            near = tuple(near)
            if T[near] == Max: #Punto no es known, se introduce en Trials
                cost = __cost( near , F[near] , Delta )
                Trials.Add( near , cost  )

    #Se ejecuta el algoritmo que calcula el valor del resto de puntos
    #EL algoritmo coge el punto x de Trials con el menor coste provisional, y lo mete en la lista de Known asignandole dicho coste como valor
    #Posteriormente actualiza el coste provisional de los puntos cercanos a x
        #Se puede demostrar, que al haber
        #A pesar de ser bastante simple, funciona debido a que la estimación de |T(x)| solo depende de los valores menores a x
            #Estas dos últimas cosas estarán explicadas con más detalle en el trabajo final, ya que son una parte importante del funcionamiento del algoritmo
    while len(Trials.Points) != 1 and Trials.Costs[1] < cutOff:
        NewKnown , NewCost = Trials.Pop() #Extrae de Trials el coste y la posición del nuevo punto
        T[ NewKnown ] = NewCost
        for NewTrial in __Next_to( list(NewKnown) , T.shape ): #Actualiza el entorno del nuevo punto
            NewTrial = tuple( NewTrial )
            if T[NewTrial] == Max: #Si el valor de NewTrial no es conocido, lo actualiza
                cost = __cost( NewTrial , F[NewTrial] , Delta )
                if Trials.Positions[NewTrial] == -1:  #Si el punto no está en Trials, lo añade
                    Trials.Add( NewTrial , cost )
                else : #Si el punto está en Trials, lo actualiza
                    Trials.Update( NewTrial , cost )
        Trials.Clean()

        #Barra de progreso
        progress += 1
        porcentaje = progress / N_Max
        if cutOff != -1 : porcentaje = max( porcentaje , NewCost / cutOff );
        print( f"{porcentaje:.2%}",  end= "\r")

    T[ T==Max ] = cutOff
    return T

#Prueba para comprobar el funcionamiento del algoritmo
#x , y = np.indices( (100,100) ) - 5
#F =  x**2 + y**2
#Gamma = np.array([ [5,5] ])
#T = solve_Eikonal(F,Gamma)
