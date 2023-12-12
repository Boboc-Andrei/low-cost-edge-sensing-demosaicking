# low-cost-edge-sensing-demosaicking

Python implementation of low cost edge-sensing demosaicking algorithm (LED) based on "Low Cost Edge Sensing for High Quality Demosaicking" (https://ieeexplore.ieee.org/document/8550686) by Y. Niu, J. Ouyang, W. Zuo and F. Wang. Initial implementation was done by the paper authors in Matlab.

I have rewritten the code in Python, so while I am not the creator of the algorithm, the Python implementation is my own work. Therefore, errors and suboptimalities can be attributed
to my own implementation, and not to the original authors.

## LED.py
applies the algorithm to all the datasets in the datasets directory and computes average time, SSIM and PSNR
saves original image and demosaicked image side by side comparison in comparison directory

##  inference.py
applies the algorithm to an input image, recieved as as command line argument or overwritten manually in default_img, then displays result.

###  References
Y. Niu, J. Ouyang, W. Zuo and F. Wang, "Low Cost Edge Sensing for High Quality Demosaicking," in IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2415-2427, May 2019, doi: 10.1109/TIP.2018.2883815.
