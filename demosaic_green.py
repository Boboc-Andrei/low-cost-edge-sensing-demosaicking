import numpy as np

def demosaic_green(mosaic, k = 0.05):
    '''
        demozaicarea planului verde prin aplicarea unei functii logistice variatiilor pe orizontala respectiv verticala
    '''
    height,width= np.shape(mosaic)

    #   adaugam margini
    p = 3
    padded = np.zeros([height+2*p,width+2*p])
    padded[p:height+p,p:width+p] = mosaic.copy()

    #   extragem canalul de verde din mozaic
    green_true = np.zeros(np.shape(padded))
    green_true[p:height+p:2,p:width+p:2] = mosaic[::2,::2]
    green_true[p+1:height+p:2,p+1:width+p:2] = mosaic[1::2,1::2]
    
    #   calculam valoarea maxima respectiv minima pentru a normaliza la final imaginea
    all_green_values = np.concatenate([mosaic[::2,::2].flatten(),mosaic[1::2,1::2].flatten()])
    g_max, g_min = np.max(all_green_values), np.min(all_green_values)

    #   initializam derivatele verticale/orizontale de ordin 1/2
    horizontal_d1 = np.zeros(np.shape(padded))
    vertical_d1 = np.zeros(np.shape(padded))
    vertical_d1 = np.zeros(np.shape(padded))

    horizontal_d2 = np.zeros(np.shape(padded))
    vertical_d2 = np.zeros(np.shape(padded))

    horizontal_mean = np.zeros(np.shape(padded))
    vertical_mean = np.zeros(np.shape(padded))


    # (1)
    #   derivata orizontala de ordin 1 in pozitiile R si B
    horizontal_d1[p:height+p, p:width+p] = (padded[p:height+p, p+1:width+p+1] - padded[p:height+p, p-1:width+p-1]) / 2

    #   derivata verticala de ordin 1 in pozitiile R si B
    vertical_d1[p:height+p, p:width+p] = (padded[p+1:height+p+1, p:width+p] - padded[p-1:height+p-1, p:width+p]) / 2

    #   derivata orizontala de ordin 2 in pozitiile R si B
    horizontal_d2[p:height+p, p:width+p] = (padded[p:height+p, p+2:width+p+2] + padded[p:height+p, p-2:width+p-2] - 2*padded[p:height+p, p:width+p]) / 4
    
    #   derivata verticala de ordin 2 in pozitiile R si B
    vertical_d2[p:height+p, p:width+p] = (padded[p+2:height+p+2, p:width+p] + padded[p-2:height+p-2, p:width+p] - 2*padded[p:height+p, p:width+p]) / 4


    #   (2)
    #   calculam variatiile pe orizontala si verticala
    horizontal_variation = np.abs(horizontal_d1) + np.abs(2*np.square(horizontal_d2))
    vertical_variation = np.abs(vertical_d1) + np.abs(2*np.square(vertical_d2))


    #   (3)
    #   calculam mediile valorilor de verde vecine pe orizontala si verticala in pozitiile R/B
    
    #   media orizontala in pozitiile R pe randuri pare
    horizontal_mean[p:height+p:2, p+1:width+p:2] = (padded[p:height+p:2, p+2:width+p+1:2] + padded[p:height+p:2, p:width+p-1:2]) / 2
    #   media orizontala in pozitiile B pe randuri impare
    horizontal_mean[p+1:height+p:2, p:width+p:2] = (padded[p+1:height+p:2, p+1:width+p+1:2] + padded[p+1:height+p:2, p-1:width+p-1:2]) / 2
    
    #   media verticala in pozitiile R pe randuri pare
    vertical_mean[p:height+p:2, p+1:width+p:2] = (padded[p+1:height+p+1:2, p+1:width+p:2] + padded[p-1:height+p-1:2, p+1:width+p:2]) / 2
    #   media verticala in pozitiile B pe randuri impare
    vertical_mean[p+1:height+p:2, p:width+p:2] = (padded[p+2:height+p+1:2, p:width+p:2] + padded[p:height+p-1:2, p:width+p:2]) / 2


    #   (13)
    #   aplicarea functiei logistice pentru determinarea valorilor de verde
    omega_h = 1 / (1 + np.exp(k*(horizontal_variation - vertical_variation)))

    #   (8)
    estimated_green = omega_h * (horizontal_mean - horizontal_d2) + (1-omega_h) * (vertical_mean - vertical_d2)

    #   suprascrierea valorilor de verde cunoscute in planul estimat
    estimated_green[p:height+p:2, p:width+p:2] = green_true[p:height+p:2,p:width+p:2]
    estimated_green[p+1:height+p:2, p+1:width+p:2] = green_true[p+1:height+p:2, p+1:width+p:2]

    estimated_green = estimated_green.clip(g_min,g_max)
    return estimated_green[p:height+p, p:width+p], omega_h[p:height+p, p:width+p]