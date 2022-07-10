import numpy as np

def DX_min( M , Delta=1 ): #Realiza la derivada parcial en la dirección x de M
    D = M[ : , 1 : , :] - M[ : , :-1 , :]
    D = D/Delta
    D = np.abs( D )
    D_M = np.zeros_like( M )
    D_M[ : , 0, :] = D[ : , 0 , :]
    D_M[ : , -1 , :] = D[ : , -1, :]
    #Para minimizar el ruido, se toma el menor de los dos
    D_M[ : , 1:-1 , :] = np.minimum(D[ : , 1 : , :] , D[ : , :-1 , :])
    return D_M

def DY_min( M , Delta=1 ): #Realiza la derivada parcial en la dirección y de M
    D = M[ 1: , :, :] - M[ :-1 , : , :]
    D = D / Delta
    D = np.abs(D)
    D_M = np.zeros_like(M)
    D_M[ 0 , :, : ] = D[ 0 , :, :]
    D_M[ -1 , : , :] = D[ -1 , : , :]
    # Para minimizar el ruido, se toma el menor de los dos
    D_M[ 1:-1 , : , :] = np.minimum(D[ 1: , :, :] , D[ :-1 , : , :])
    return D_M

def DZ_min( M , Delta=1 ): #Realiza la derivada parcial en la dirección y de M
    D = M[ : , :, 1:] - M[ : , : , :-1]
    D = D / Delta
    D = np.abs(D)
    D_M = np.zeros_like(M)
    D_M[ : , : , 0] = D[ : , : , 0]
    D_M[ : , : , -1 ] = D[ : , : , -1 ]
    # Para minimizar el ruido, se toma el menor de los dos
    D_M[ : , : , 1:-1 ] = np.minimum(D[ : , : , 1:] , D[ : , : , :-1 ])
    return D_M

def potencial3D( Imag , Delta = 1):
    #Transforma una imagen en la matriz portencial
    Pot = np.sqrt( DY_min( Imag , Delta = Delta )**2 + DX_min( Imag , Delta = Delta )**2 + DZ_min( Imag , Delta = Delta )**2)
    return Pot