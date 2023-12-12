import numpy as np

def demosaic_red(mosaic, greens, omega_h, k=0.05):
    '''
        demozaicarea planului rosu se face prin estimarea planului (G-R) si scaderea acestuia din planul G

        in prima etapa, estimam (G-R) in pozitiile B folosind derivatele pe directiile diagonale ale mozaicului original,
        folosind o functie logistica asemanatoare cu cea de la planul G

        in etapa a doua, estimam G-R in pozitiile G folosindu-ne de valorile G-R cunoscute, impreuna cu cele
        calculate in etapa precedenta, folosind aceeasi functie logistica ca la planul G
    '''
    height, width = np.shape(mosaic)
    
    #   adaugam margini pentru a calcula derivatele in punctele fara informatie de rosu
    p = 3
    padded = np.zeros([height+2*p, width+2*p])
    padded[p:height+p, p:width+p] = mosaic.copy()

    padded_green = np.zeros(np.shape(padded))
    padded_green[p:height+p, p:width+p] = greens

    padded_omega = np.zeros(np.shape(padded))
    padded_omega[p:height+p, p:width+p] = omega_h

    #   extragem canalul de rosu din mozaic
    red_true = np.zeros(np.shape(padded))
    red_true[p:height+p:2, p+1:width+p:2] = mosaic[::2, 1::2]

    #   calculam valorile maxime si minime pentru a normaliza la final
    r_max,r_min = np.max(mosaic[::2, 1::2]), np.min(mosaic[::2, 1::2])

    #   initializam derivatele diagonalei principale si secundare de ordin 1 si 2 in pozitiile B
    main_diag_d1 = np.zeros(np.shape(padded))
    main_diag_d2 = np.zeros(np.shape(padded))
    second_diag_d1 = np.zeros(np.shape(padded))
    second_diag_d2 = np.zeros(np.shape(padded))

    GR_main_diag_mean = np.zeros(np.shape(padded))
    GR_second_diag_mean = np.zeros(np.shape(padded))
    GB_main_diag_d2 = np.zeros(np.shape(padded))
    GB_second_diag_d2 = np.zeros(np.shape(padded))
    
    GR_horizontal_mean_at_G = np.zeros(np.shape(padded))
    GR_vertical_mean_at_G = np.zeros(np.shape(padded))
    estimated_GR = np.zeros(np.shape(padded))
    estim_GR_horizontal_d1_at_nonG = np.zeros(np.shape(padded))
    estim_GR_vertical_d1_at_nonG = np.zeros(np.shape(padded))
    estim_GR_horizontal_d2_at_G = np.zeros(np.shape(padded))
    estim_GR_vertical_d2_at_G = np.zeros(np.shape(padded))


    #   PARTEA 1: estimarea planului G-R in pozitiile B

    #   (15)
    #   derivata de ordin 1 a diagonalei principale in pozitiile B
    main_diag_d1[p+1:height+p:2, p:width+p:2] = (padded[p+2:height+p+1:2, p+1:width+p+1:2] - padded[p:height+p-1:2, p-1:width+p-1:2]) / 2**1.5
    #   derivata de ordin 1 a diagonalei secundare in pozitiile B
    second_diag_d1[p+1:height+p:2, p:width+p:2] = (padded[p:height+p-1:2, p+1:width+p+1:2] - padded[p+2:height+p+1:2, p-1:width+p-1:2]) / 2**1.5
    #   derivata de ordin 2 a diagonalei principale in pozitiile B
    main_diag_d2[p+1:height+p:2, p:width+p:2] = (padded[p+3:height+p+2:2, p+2:width+p+2:2] + padded[p-1:height+p-2:2, p-2:width+p-2:2] - 2*padded[p+1:height+p:2, p:width+p:2]) / 8
    #   derivata de ordin 2 a diagonalei secundare in pozitiile B
    second_diag_d2[p+1:height+p:2, p:width+p:2] = (padded[p-1:height+p-2:2, p+2:width+p+2:2] + padded[p+3:height+p+2:2, p-2:width+p-2:2] - 2*padded[p+1:height+p:2, p:width+p:2]) / 8
    
    #   (16)
    #   calculul variatiilor pe diagonala secundara respectiv principala
    main_diag_variation = np.abs(main_diag_d1) + np.abs(2**1.5 * main_diag_d2)
    second_diag_variation = np.abs(second_diag_d1) + np.abs(2**1.5 * second_diag_d2)

    #   (17)
    omega_md = 1 / (1 + np.abs(k*(main_diag_variation - second_diag_variation)))

    #   (18)
    #   planul diferenta intre verde si R adevarat respectiv B adevarat - il folosim pentru a calcula mediile dintre valorile G^-R vecine pe diagonale in pozitiile B
    #                                                                     si derivatele de ordin 2 pe diagonale a planului G^-B
    G_diff = padded_green - padded
    GR_main_diag_mean[p+1:height+p:2, p:width+p:2] = (G_diff[p+2:height+p+1:2, p+1:width+p+1:2] + G_diff[p:height+p-1:2, p-1:width+p-1:2]) / 2
    GR_second_diag_mean[p+1:height+p:2, p:width+p:2] = (G_diff[p:height+p-1:2, p+1:width+p+1:2] + G_diff[p+2:height+p+1:2, p-1:width+p-1:2]) / 2

    #   (19)
    #   derivata de ordin 2 pe diagonala principala a planului G^-B

    GB_main_diag_d2[p+1:height+p:2, p:width+p:2] = (G_diff[p+3:height+p+2:2, p+2:width+p+2:2] + G_diff[p-1:height+p-2:2, p-2:width+p-2:2] - 2*G_diff[p+1:height+p:2, p:width+p:2]) / 8
    GB_second_diag_d2[p+1:height+p:2, p:width+p:2] = (G_diff[p+3:height+p+2:2, p-2:width+p-2:2] + G_diff[p-1:height+p-2:2, p+2:width+p+2:2] - 2*G_diff[p+1:height+p:2, p:width+p:2]) / 8

    #   (20)
    #   functie logistica de estimare a G-R in pozitiile B
    estimated_GR_at_B = omega_md * (GR_main_diag_mean - GB_main_diag_d2) + (1-omega_md) * (GR_second_diag_mean - GB_second_diag_d2)

    # PARTEA 2 : estimarea planului G-R in pozitiile G

    #   (22)
    #   media pe orizontala a planului (G-R) in pozitiile G pe randuri pare
    GR_horizontal_mean_at_G[p:height+p:2, p:width+p:2] = (G_diff[p:height+p:2, p-1:width+p-1:2] + G_diff[p:height+p:2, p+1:width+p+1:2]) / 2
    #   media pe orizontala a planului (G-R) in pozitiile G pe randuri impare
    GR_horizontal_mean_at_G[p+1:height+p:2, p+1:width+p:2] = (estimated_GR_at_B[p+1:height+p:2, p:width+p-1:2] + estimated_GR_at_B[p+1:height+p:2, p+2:width+p+1:2]) / 2

    #   media pe verticala a planului (G-R) in pozitiile G pe randuri pare
    GR_vertical_mean_at_G[p:height+p:2, p:width+p:2] = (estimated_GR_at_B[p+1:height+p+1:2, p:width+p:2] + estimated_GR_at_B[p-1:height+p-1:2, p:width+p:2]) / 2
    #   media pe orizontala a planului (G-R) in pozitiile G pe randuri impare
    GR_vertical_mean_at_G[p+1:height+p:2, p+1:width+p:2] = (G_diff[p:height+p-1:2, p+1:width+p:2] + G_diff[p+2:height+p+1:2, p+1:width+p:2]) / 2

    #   (24)
    #   calculam derivatele de ordin 1 ale planmului (G-R) pe orizontala/verticala in pozitiile non-G

    #   actualizam (G-R) cu pozitiile B calculate anterior
    estimated_GR[p+1:height+p:2, p:width+p:2] = estimated_GR_at_B[p+1:height+p:2, p:width+p:2]
    #   actualizam (G-R) cu pozitiile (Gestimat - Rreal) cunoscute
    estimated_GR[p:height+p:2, p+1:width+p:2] = G_diff[p:height+p:2, p+1:width+p:2]

    #   derivata de ord. 1 a (G-R) pe orizontala pe randuri pare in pozitii R
    estim_GR_horizontal_d1_at_nonG[p:height+p:2, p-1:width+p+1:2] = (estimated_GR[p:height+p:2, p+1:width+p+3:2] - estimated_GR[p:height+p:2, p-3:width+p-1:2]) / 4
    #   derivata de ord.1 a (G-R) pe orizontala pe randuri impare in pozitii B
    estim_GR_horizontal_d1_at_nonG[p+1:height+p:2, p:width+p+1:2] = (estimated_GR[p+1:height+p:2, p+2:width+p+3:2] - estimated_GR[p+1:height+p:2, p-2:width+p-1:2]) / 4

    #   derivata de ord. 1 a (G-R) pe verticala pe randuri pare in pozitii R
    estim_GR_vertical_d1_at_nonG[p:height+p+1:2, p+1:width+p:2] = (estimated_GR[p+2:height+p+3:2, p+1:width+p:2] - estimated_GR[p-2:height+p-1:2, p+1:width+p:2]) / 4
    #   derivata de ord. 1 a (G-R) pe verticala pe randuri impare in pozitii B
    estim_GR_vertical_d1_at_nonG[p+1:height+p+1:2, p:width+p:2] = (estimated_GR[p+3:height+p+3:2, p:width+p:2] - estimated_GR[p-1:height+p-1:2, p:width+p:2]) / 4

    #   (23)
    #   calculam derivata de ordin 2 a (G-R) in pozitii G
    estim_GR_horizontal_d2_at_G = np.zeros(np.shape(padded))
    estim_GR_horizontal_d2_at_G[p:height+p, p:width+p] = (estim_GR_horizontal_d1_at_nonG[p:height+p, p+1:width+p+1] - estim_GR_horizontal_d1_at_nonG[p:height+p, p-1:width+p-1]) / 2
    estim_GR_vertical_d2_at_G[p:height+p, p:width+p] = (estim_GR_vertical_d1_at_nonG[p+1:height+p+1, p:width+p] - estim_GR_vertical_d1_at_nonG[p-1:height+p-1, p:width+p]) / 2

    #   calculam cu functia logistica (G-R) in pozitiile G
    estimated_GR = padded_omega * (GR_horizontal_mean_at_G - estim_GR_horizontal_d2_at_G) + (1-padded_omega) * (GR_vertical_mean_at_G - estim_GR_vertical_d2_at_G)
    #   suprascriem (G-R) in pozitiile B calculate anterior
    estimated_GR[p+1:height+p:2, p:width+p:2] = estimated_GR_at_B[p+1:height+p:2, p:width+p:2]
    #   suprascriem (G-R) cu pozitiile (Gestimat - Rreal) cunoscute
    estimated_GR[p:height+p:2, p+1:width+p:2] = G_diff[p:height+p:2, p+1:width+p:2]

    #   (21)
    #   scadem din planul G^ planul diferenta G-R estimat pentru a obtine valorile de rosu
    estimated_red = padded_green - estimated_GR

    estimated_red = estimated_red.clip(r_min,r_max)

    return estimated_red[p:height+p, p:width+p]