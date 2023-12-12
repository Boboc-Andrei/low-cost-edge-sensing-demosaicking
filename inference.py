import matplotlib.pyplot as plt
import numpy as np
from LED import LED_demosaicking, generate_bayer
import skimage.io as io
import time


def show_demosaicked_channels(img):
    mosaic,raw = generate_bayer(img)

    reconstructed_img = LED_demosaicking(mosaic)

    reconstructed_red = reconstructed_img[:,:,0]
    reconstructed_green = reconstructed_img[:,:,1]
    reconstructed_blue = reconstructed_img[:,:,2]


    greens = np.zeros(np.shape(img))
    greens[:,:,1] = reconstructed_green
    reds = np.zeros(np.shape(img))
    reds[:,:,0] = reconstructed_red
    blues = np.zeros(np.shape(img))
    blues[:,:,2] = reconstructed_blue


    plt.figure()

    plt.subplot(2,3,1)
    plt.title('Original image')
    plt.axis('off')
    plt.imshow(img)


    plt.subplot(2,3,2)
    plt.title('Bayer pattern image')
    plt.axis('off')
    plt.imshow(raw)


    plt.subplot(2,3,3)
    plt.title('Reconstructed image')
    plt.axis('off')
    plt.imshow(reconstructed_img)


    plt.subplot(2,3,4)
    plt.title('Interpolated red channel')
    plt.axis('off')
    plt.imshow(reds)


    plt.subplot(2,3,5)
    plt.title('Interpolated green channel')
    plt.axis('off')
    plt.imshow(greens)


    plt.subplot(2,3,6)
    plt.title('Interpolated blue channel')
    plt.axis('off')
    plt.imshow(blues)
    plt.tight_layout()


#   imaginea input
test_img = '1.tif'


img = io.imread(test_img)/255

mosaic,raw = generate_bayer(img)


start = time.time()
reconstructed_img = LED_demosaicking(mosaic)
end = time.time()

reconstructed_red = reconstructed_img[:,:,0]
reconstructed_green = reconstructed_img[:,:,1]
reconstructed_blue = reconstructed_img[:,:,2]

t = end-start
print(f'Finished in {t:.3f} seconds')

show_demosaicked_channels(img)

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