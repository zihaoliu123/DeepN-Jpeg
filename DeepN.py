import numpy as np
from numpy import pi
import math
import tensorflow as tf
from scipy.fftpack import dct, idct, rfft, irfft
from keras.preprocessing import image
T = np.array([
        [0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
        [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
        [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
        [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
        [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
        [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
        [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
        [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]
    ])

""
Jpeg_def_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 36, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
])

""
num = 8


def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
def rfft2 (block):
    return rfft(rfft(block.T).T)
def irfft2(block):
    return irfft(irfft(block.T).T)


def main(argv=None):
    img = image.load_img('./a.JPEG')
    input = image.img_to_array(img)
    input_matrix = input[0:224,0:224]
    input_matrix = np.expand_dims(input_matrix, axis=0)
    n = input_matrix.shape[0]
    h = input_matrix.shape[1]
    w = input_matrix.shape[2]
    c = input_matrix.shape[3]
    horizontal_blocks_num = w / num
    output2=np.zeros((c,h, w))
    output3=np.zeros((n,3,h, w))    
    vertical_blocks_num = h / num
    coeff_array = np.zeros((8,8))
    n_block = np.split(input_matrix,n,axis=0)
# 1. Frequency analysis
    for i in range(0, n):
        c_block = np.split(n_block[i],c,axis =3)
        for ch_block in c_block:
            vertical_blocks = np.split(ch_block, vertical_blocks_num,axis = 1)
            for block_ver in vertical_blocks:
                hor_blocks = np.split(block_ver,horizontal_blocks_num,axis = 2)
                for block in hor_blocks:
                    block = np.reshape(block,(num,num))
                    block = dct2(block)
                    coeff_array = np.dstack((coeff_array, block))
# 2. Generate Important frequency component index
    imp_index = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            imp_index[i][j] = np.std(coeff_array[i][j][:])
    imp_index = np.round(imp_index)
    print(imp_index)


if __name__ == '__main__':
    main()
