import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import color
import time
import cv2
from demosaic_green import demosaic_green
from demosaic_red import demosaic_red
from demosaic_blue import demosaic_blue
import os
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def generate_bayer(img):
    '''
        generate bayer pattern:\n
        G R G R\n
        B G B G
    '''
    width,height,_ = np.shape(img)

    red = np.zeros([width,height])
    green = np.zeros([width,height])
    blue = np.zeros([width,height])

    red[0::2,1::2] = img[0::2,1::2,0]
    green[0::2,0::2] = img[0::2,0::2,1]
    green[1::2,1::2] = img[1::2,1::2,1]
    blue[1::2,0::2] = img[1::2,0::2,2]
    bayer_pattern = np.array([red,green,blue]).transpose([1,2,0])
    return red+green+blue, bayer_pattern


def LED_demosaicking(mosaic):
    '''
        demozaicarea secventiala a planului verde, si ulterior a planelor rosu si albastru folosind planele 
        de diferenta G-R respectiv G-B.
        implementarea demozaicarii planului B este identica cu cea a planului R, avand doar anumiti indecsi modificati
    '''
    height, width = np.shape(mosaic)

    reconstructed_green, omega_h = demosaic_green(mosaic)
    reconstructed_red = demosaic_red(mosaic,reconstructed_green,omega_h)
    reconstructed_blue = demosaic_blue(mosaic,reconstructed_green,omega_h)

    reconstructed_img = np.zeros([height,width,3])
    reconstructed_img[:,:,0] = reconstructed_red
    reconstructed_img[:,:,1] = reconstructed_green
    reconstructed_img[:,:,2] = reconstructed_blue
    return reconstructed_img


if __name__ == '__main__':


    datasets = os.listdir('datasets')
    metrics = {}

    for dataset in datasets:
        dataset_path = os.path.join('datasets',dataset)
        images = os.listdir(dataset_path)

        times = []
        PSNRs = []
        SSIMs = []
        
        for i, image in enumerate(images):
            image_path = os.path.join(dataset_path, image)
            img = io.imread(image_path)/255
            height,width,_ = np.shape(img)

            mosaic, raw = generate_bayer(img)
            start = time.time()
            reconstructed_img = LED_demosaicking(mosaic)
            end = time.time()
            
            p=4
            img = img[p:height-p, p:width-p,:]
            reconstructed_img = reconstructed_img[p:height-p, p:width-p,:]
            height-=2*p
            
            times.append(end-start)
            PSNRs.append(PSNR(img,reconstructed_img))
            SSIMs.append(SSIM(img,reconstructed_img, channel_axis = 2, data_range=1))

            separator = np.zeros([height,10,3])

            side_by_side = np.concatenate([img,separator,reconstructed_img], axis=1)
            side_by_side = np.uint8(side_by_side * 255)
            side_by_side_path = os.path.join('comparison', dataset+str(i).zfill(2)+'.jpg')
            io.imsave(side_by_side_path,side_by_side)

        metrics[dataset] = {
            'time': np.mean(times),
            'PSNR': np.mean(PSNRs),
            'SSIM': np.mean(SSIMs)
        }

    for dataset in metrics:
        print(f'{dataset}: ')
        for metric, value in metrics[dataset].items():
            print(f'  {metric}: {value:.3f}')


    img = io.imread('1.tif')/255
    mosaic, raw = generate_bayer(img)
    reconstructed_img = LED_demosaicking(mosaic)

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Original image')
    plt.axis('off')
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.title('Demosaicked image')
    plt.axis('off')
    plt.imshow(reconstructed_img)

    plt.tight_layout()
    plt.show()
