import numpy as np

def __D( a ,b , Delta ):
    return (b-a)/Delta

def __MapedVector( Vector , Map ):
    #Prepara un vector para ser introducido en la función derivada
    MapedVector = Vector.copy()
    Map = Map.astype(int)
    MapedVector[ Map!= -1 ] = Vector[ Map[Map != -1] ]
    return MapedVector

def DX( M , Delta=1 ): #Realiza la derivada parcial en la dirección x de una matriz M
    D = M[ : , 1 : ] - M[ : , :-1 ]
    D = D/Delta
    D_M = np.zeros_like( M )
    D_M[ : , 0] = D[ : , 0 ]
    D_M[ : , -1 ] = D[ : , -1]
    D_M[ : , 1:-1 ] = (D[ : , 1 : ] + D[ : , :-1 ]) / 2
    return D_M

def DY( M , Delta=1 ): #Realiza la derivada parcial en la dirección y de una matriz M
    D = M[ 1: , :] - M[ :-1 , : ]
    D = D / Delta
    D_M = np.zeros_like(M)
    D_M[ 0 , : ] = D[ 0 , :]
    D_M[ -1 , : ] = D[ -1 , : ]
    D_M[ 1:-1 , : ] = (D[ 1: , :] + D[ :-1 , : ]) / 2
    return D_M

def DXBand( Band , Delta = 1): #Derivada en X de la banda
    BandX1 = __D( Band , __MapedVector(Band,MapUp) , Delta)
    BandX2 = __D(__MapedVector(Band, MapDown), Band, Delta)
    return ( BandX1 + BandX2) / 2

def DYBand( Band , Delta = 1 ): #Derivada en Y de la banda
    BandY1 = __D(Band, __MapedVector(Band, MapRight), Delta)
    BandY2 = __D(__MapedVector(Band, MapLeft), Band, Delta)
    return (BandY1 + BandY2) / 2


def Band_Propagation( Phi0 , P = 0 , a = 0 , b = 1000 , Deltat = 0.01 , Deltax = 1 , steps = 0 , eps = 0.5 , beta = 1 , c = 1):
    """
    Propaga la función Phi mediante la ecuación Phi_t + g(1-epsK)|VPhi| - bet VP VPhi
    Se limita la propagación a los puntos que en Phi0 tienen un valor entre a y b
    """
    #Parámetros beta , esp

    Phi = Phi0[ (Phi0>a) & (Phi0<b) ] #Se extrae una lista con los valores que toma Phi0 en la banda
    #Antes de realizar operaciones, es necesario obtener una serie de mapeos
    #Se mapea la banda a sus posiciones en la matriz original
    X , Y = np.indices( Phi0.shape )
    PosX = X[ (a<Phi0) & (Phi0<b) ]
    PosY = Y[ (a<Phi0) & (Phi0<b) ]
    #Se realiza también el mapeado inverso, cada punto de la matriz a su posición en la banda, si no está, toma el valor -1
    Positions = -1 * np.ones_like(Phi0)
    Positions[ PosX , PosY ] = np.arange( len(Phi) )

    #Se mapea cada punto en la banda con los puntos en la banda que lo rodean
    #La siguiente parte se hará de la forma más comprimida posible, se ha comprobado a parte que funciona
    RefShape = np.array(Phi0.shape) + np.array([2,2])
    Ref = -1 * np.ones( RefShape )
    Ref[ 1:-1 , 1:-1 ] = Positions
    global MapUp, MapDown, MapLeft, MapRight
    MapUp = Ref[ PosX , PosY+1 ]
    MapDown = Ref[ PosX+2 , PosY+1 ]
    MapLeft = Ref[ PosX+1 , PosY ]
    MapRight = Ref[ PosX+1 , PosY+2 ]


    #Prepara el vector P para las operaciones
    #Como estas operaciones solo se tienen que hacer una vez, se obtienen las derivadas de P en toda la matriz
    PX = DX( P , Deltax )
    PY = DY( P , Deltax )
    F = c / ( 1 + P )#Función velocidad usada en el FM
    #Se restringen los resultados obtenidos a la banda
    PX = PX[ (Phi0>a) & (Phi0<b) ]
    PY = PY[ (Phi0>a) & (Phi0<b) ]
    F = F[ (Phi0>a) & (Phi0<b) ]

    #Se avanza la función
    for i in range(steps):
        #Se obtienen las derivadas de Phi
        PhiX = DXBand( Phi , Deltax )
        PhiY = DYBand( Phi , Deltax )
        PhiXX = DXBand( PhiX , Deltax )
        PhiYY = DYBand( PhiY , Deltax )
        PhiXY = (DXBand( PhiY , Deltax) + DYBand( PhiX , Deltax )) / 2
        #Se obtiene la curvatura de Phi
        K = (PhiXX * PhiY ** 2 - 2 * PhiX * PhiY * PhiXY + PhiYY * PhiX ** 2) / (PhiX ** 2 + PhiY ** 2 + 0.0001) ** (3 / 2)
        H = np.zeros_like( Phi0 )
        H[ (a<Phi0) & (Phi0<b) ] = PhiX

        #El algoritmo se reduce a aplicar la siguiente función de manera repetida
        #El resto del código está para poder aplicar la función al intervalo [a,b] y no a toda la imágen, mejorando mucho la eficiencia
        Phi = Phi + Deltat*(   -F*(1-eps*K) * np.sqrt( PhiX**2 + PhiY**2 ) - beta*( np.dot(PhiX,PX) + np.dot(PhiY,PY) ) )

    Phi0[PosX,PosY] = Phi
    Phi0[ Phi0 < a ] = a
    Phi0[ Phi0 > b ] = b

    return Phi0




