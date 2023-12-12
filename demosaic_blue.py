import numpy as np

def demosaic_blue(mosaic, greens, omega_h, k=0.05):
    '''
        demozaicarea planului albastru urmareste acelasi algoritm ca si al planului rosu,
        inlocuind rosu cu albastru si vice-versa
    '''
    height, width = np.shape(mosaic)

    p=3
    padded = np.zeros([height+2*p, width+2*p])
    padded[p:height+p, p:width+p] = mosaic.copy()

    padded_green = np.zeros(np.shape(padded))
    padded_green[p:height+p, p:width+p] = greens

    padded_omega = np.zeros(np.shape(padded))
    padded_omega[p:height+p, p:width+p] = omega_h

    #   extragem canalul de albastru din mozaic
    blue_true = np.zeros(np.shape(padded))
    blue_true[p+1:height+p:2, p:width+p:2] = mosaic[1::2, ::2]

    #   calculam valorile maxime si minime pentru a normaliza la final
    b_max,b_min = np.max(mosaic[1::2, ::2]), np.min(mosaic[1::2, ::2])

    #   initializam derivatele diagonalei principale si secundare de ordin 1 si 2 in pozitiile B
    main_diag_d1 = np.zeros(np.shape(padded))
    main_diag_d2 = np.zeros(np.shape(padded))
    second_diag_d1 = np.zeros(np.shape(padded))
    second_diag_d2 = np.zeros(np.shape(padded))

    GB_main_diag_mean = np.zeros(np.shape(padded))
    GB_second_diag_mean = np.zeros(np.shape(padded))
    GR_main_diag_d2 = np.zeros(np.shape(padded))
    GR_second_diag_d2 = np.zeros(np.shape(padded))
    
    GB_horizontal_mean_at_G = np.zeros(np.shape(padded))
    GB_vertical_mean_at_G = np.zeros(np.shape(padded))
    estimated_GB = np.zeros(np.shape(padded))
    estim_GB_horizontal_d1_at_nonG = np.zeros(np.shape(padded))
    estim_GB_vertical_d1_at_nonG = np.zeros(np.shape(padded))
    estim_GB_horizontal_d2_at_G = np.zeros(np.shape(padded))
    estim_GB_vertical_d2_at_G = np.zeros(np.shape(padded))


    #   PARTEA 1: estimarea planului G-B in pozitiile R

    #   (15)
    #   derivata de ordin 1 a diagonalei principale in pozitiile R
    main_diag_d1[p:height+p:2, p+1:width+p:2] = (padded[p+1:height+p+1:2, p+2:width+p+1:2] - padded[p-1:height+p-1:2, p:width+p-1:2]) / 2**1.5
    #   derivata de ordin 1 a diagonalei secundare in pozitiile R
    second_diag_d1[p:height+p:2, p+1:width+p:2] = (padded[p-1:height+p-1:2, p+2:width+p+1:2] - padded[p+1:height+p+1:2, p:width+p-1:2]) / 2**1.5
    #   derivata de ordin 2 a diagonalei principale in pozitiile R
    main_diag_d2[p:height+p:2, p+1:width+p:2] = (padded[p+2:height+p+2:2, p+3:width+p+2:2] + padded[p-2:height+p-2:2, p-1:width+p-2:2] - 2*padded[p:height+p:2, p+1:width+p:2]) / 8
    #   derivata de ordin 2 a diagonalei secundare in pozitiile R
    second_diag_d2[p:height+p:2, p+1:width+p:2] = (padded[p-2:height+p-2:2, p+3:width+p+2:2] + padded[p+2:height+p+2:2, p-1:width+p-1:2] - 2*padded[p:height+p:2, p+1:width+p:2]) / 8
    
    #   (16)
    #   calculul variatiilor pe diagonala secundara respectiv principala
    main_diag_variation = np.abs(main_diag_d1) + np.abs(2**1.5 * main_diag_d2)
    second_diag_variation = np.abs(second_diag_d1) + np.abs(2**1.5 * second_diag_d2)
    
    #   (17)
    omega_md = 1 / (1 + np.abs(k*(main_diag_variation - second_diag_variation)))

    #   (18)
    #   planul diferenta intre verde si B adevarat respectiv R adevarat - il folosim pentru a calcula mediile dintre valorile G^-B vecine pe diagonale in pozitiile R
    #                                                                     si derivatele de ordin 2 pe diagonale a planului G^-R
    G_diff = padded_green - padded
    GB_main_diag_mean[p:height+p:2, p+1:width+p:2] = (G_diff[p+1:height+p+1:2, p+2:width+p+1:2] + G_diff[p-1:height+p-1:2, p:width+p-1:2]) / 2**1.5
    GB_second_diag_mean[p:height+p:2, p+1:width+p:2] = (G_diff[p-1:height+p-1:2, p+2:width+p+1:2] + G_diff[p+1:height+p+1:2, p:width+p-1:2]) / 2**1.5
    
    #   (19)
    #   derivata de ordin 2 pe diagonala principala a planului G^-R
                                                                                                                                         
    GR_main_diag_d2[p:height+p:2, p+1:width+p:2] = (G_diff[p+2:height+p+2:2, p+3:width+p+2:2] + G_diff[p-2:height+p-2:2, p-1:width+p-2:2] - 2*G_diff[p:height+p:2, p+1:width+p:2]) / 8
    GR_second_diag_d2[p:height+p:2, p+1:width+p:2] = (G_diff[p+2:height+p+2:2, p-1:width+p-2:2] + G_diff[p-2:height+p-2:2, p+3:width+p+2:2] - 2*G_diff[p:height+p:2, p+1:width+p:2]) / 8

    #   (20)
    #   functie logistica de estimare a G-R in pozitiile B
    estimated_GB_at_R = omega_md * (GB_main_diag_mean - GR_main_diag_d2) + (1-omega_md) * (GB_second_diag_mean - GR_second_diag_d2)

    # PARTEA 2 : estimarea planului G-B in pozitiile G

    #   (22)
    #   media pe orizontala a planului (G-B) in pozitiile G pe randuri pare
    GB_horizontal_mean_at_G[p:height+p:2, p:width+p:2] = (estimated_GB_at_R[p:height+p:2, p-1:width+p-1:2] + estimated_GB_at_R[p:height+p:2, p+1:width+p+1:2]) / 2
    #   media pe orizontala a planului (G-B) in pozitiile G pe randuri impare
    GB_horizontal_mean_at_G[p+1:height+p:2, p+1:width+p:2] = (G_diff[p+1:height+p:2, p:width+p-1:2] + G_diff[p+1:height+p:2, p+2:width+p+1:2]) / 2

    #   media pe verticala a planului (G-B) in pozitiile G pe randuri pare
    GB_vertical_mean_at_G[p:height+p:2, p:width+p:2] = (G_diff[p+1:height+p+1:2, p:width+p:2] + G_diff[p-1:height+p-1:2, p:width+p:2]) / 2
    #   media pe orizontala a planului (G-B) in pozitiile G pe randuri impare
    GB_vertical_mean_at_G[p+1:height+p:2, p+1:width+p:2] = (estimated_GB_at_R[p:height+p-1:2, p+1:width+p:2] + estimated_GB_at_R[p+2:height+p+1:2, p+1:width+p:2]) / 2

    #   (24)
    #   calculam derivatele de ordin 1 ale planmului (G-R) pe orizontala/verticala in pozitiile non-G

    #   actualizam (G-B) in pozitiile R calculate anterior
    estimated_GB[p+1:height+p:2, p:width+p:2] = G_diff[p+1:height+p:2, p:width+p:2]
    #   actualizam (G-B) cu pozitiile (Gestimat - Breal) cunoscute
    estimated_GB[p:height+p:2, p+1:width+p:2] = estimated_GB_at_R[p:height+p:2, p+1:width+p:2]

    #   derivata de ord. 1 a (G-B) pe orizontala pe randuri pare in pozitii R
    estim_GB_horizontal_d1_at_nonG[p:height+p:2, p-1:width+p+1:2] = (estimated_GB[p:height+p:2, p+1:width+p+3:2] - estimated_GB[p:height+p:2, p-3:width+p-1:2]) / 4
    #   derivata de ord.1 a (G-B) pe orizontala pe randuri impare in pozitii B
    estim_GB_horizontal_d1_at_nonG[p+1:height+p:2, p:width+p+1:2] = (estimated_GB[p+1:height+p:2, p+2:width+p+3:2] - estimated_GB[p+1:height+p:2, p-2:width+p-1:2]) / 4

    #   derivata de ord. 1 a (G-B) pe verticala pe randuri pare in pozitii R
    estim_GB_vertical_d1_at_nonG[p:height+p+1:2, p+1:width+p:2] = (estimated_GB[p+2:height+p+3:2, p+1:width+p:2] - estimated_GB[p-2:height+p-1:2, p+1:width+p:2]) / 4
    #   derivata de ord. 1 a (G-B) pe verticala pe randuri impare in pozitii B
    estim_GB_vertical_d1_at_nonG[p+1:height+p+1:2, p:width+p:2] = (estimated_GB[p+3:height+p+3:2, p:width+p:2] - estimated_GB[p-1:height+p-1:2, p:width+p:2]) / 4

    #   (23)
    #   calculam derivata de ordin 2 a (G-B) in pozitii G
    estim_GB_horizontal_d2_at_G = np.zeros(np.shape(padded))
    estim_GB_horizontal_d2_at_G[p:height+p, p:width+p] = (estim_GB_horizontal_d1_at_nonG[p:height+p, p+1:width+p+1] - estim_GB_horizontal_d1_at_nonG[p:height+p, p-1:width+p-1]) / 2
    estim_GB_vertical_d2_at_G[p:height+p, p:width+p] = (estim_GB_vertical_d1_at_nonG[p+1:height+p+1, p:width+p] - estim_GB_vertical_d1_at_nonG[p-1:height+p-1, p:width+p]) / 2
    
    #   calculam cu functia logistica (G-B) in pozitiile G
    estimated_GB = padded_omega * (GB_horizontal_mean_at_G - estim_GB_horizontal_d2_at_G) + (1-padded_omega) * (GB_vertical_mean_at_G - estim_GB_vertical_d2_at_G)
    #   suprascriem (G-B) in pozitiile R calculate anterior
    estimated_GB[p:height+p:2, p+1:width+p:2] = estimated_GB_at_R[p:height+p:2, p+1:width+p:2]
    #   suprascriem (G-B) cu pozitiile (Gestimat - Breal) cunoscute
    estimated_GB[p+1:height+p:2, p:width+p:2] = G_diff[p+1:height+p:2, p:width+p:2]

    #   (21)
    #   scadem din planul G^ planul diferenta G-B estimat pentru a obtine valorile de albastru
    estimated_blue = padded_green - estimated_GB

    estimated_blue = estimated_blue.clip(b_min,b_max)

    return estimated_blue[p:height+p, p:width+p]