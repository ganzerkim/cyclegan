# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:21:53 2020

@author: MG
cycle-GAN-Kaggle
source code link : https://www.kaggle.com/soodoshll/cyclegan
"""
#%%
import  os,shutil


import keras.backend as K
from keras.models import Model

from keras.layers import Conv2D, BatchNormalization, Input, Dropout, Add
from keras.layers import Conv2DTranspose, Reshape, Activation
from keras.layers import Concatenate

from keras.optimizers import Adam

from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu,tanh

from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np

import glob
import random

import matplotlib.pyplot as plt
import scipy.misc

from keras.utils import multi_gpu_model

#%%
def load_image(fn, image_size):
    """
    이미지 로드
    fn:이미지 파일 경로
    """
    # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
    # 이미지 로드 및 RGB형식으로 변환
    im = load_img(fn)
    
    # 이미지 가운데 정사각형으로 Cropping
    if (im.size[0] >= im.size[1]):
        im = im.crop(((im.size[0] - im.size[1])//2, 0, (im.size[0] + im.size[1])//2, im.size[1]))
    else:
        im = im.crop((0, (im.size[1] - im.size[0])//2, im.size[0], (im.size[0] + im.size[1])//2))
    #crop(start_x, start_y, start_x + width, start_y + width)
    
    # resize 
    im = im.resize((image_size, image_size))
    
    # -1 ~ 1 normalization
    arr = img_to_array(im) / 255 * 2 - 1   
    
    return arr

#%%
class DataSet(object):
    """
    데이터 관리를 위한 클래스
    """
    def __init__(self, data_path, image_size = 256):
                
        # dataset 경로
        self.data_path = data_path
        self.epoch = 0
        #데이터리스트 초기화 (자체 메서드 호출)
        self.__init_list()
        self.image_size = image_size
        
    def __init_list(self):
        # https://docs.python.org/3/library/glob.html
        self.data_list = glob.glob(self.data_path)        
        "../trainA/*.png"
    
        # https://docs.python.org/3/library/random.html#random.shuffle
        random.shuffle(self.data_list)
        
        # 포인터 초기화
        self.ptr = 0
        
    def get_batch(self, batchsize):
      
        if (self.ptr + batchsize >= len(self.data_list)):
            batch = [load_image(x, self.image_size) for x in self.data_list[self.ptr:]]
            rest = self.ptr + batchsize - len(self.data_list)
            self.__init_list()
            batch.extend([load_image(x, self.image_size) for x in self.data_list[:rest]])
            self.ptr = rest
            self.epoch += 1
        else:
            batch = [load_image(x, self.image_size) for x in self.data_list[self.ptr:self.ptr + batchsize]]
            self.ptr += batchsize
        
        return self.epoch, np.array(batch)
        
    def get_pics(self, num):
     
        return np.array([load_image(x, self.image_size) for x in random.sample(self.data_list, num)])
#%%
def arr2image(X):
    int_X = (( X + 1) / 2 * 255).clip(0, 255).astype('uint8')
    return array_to_img(int_X)

def generate(img, fn):
    r = fn([np.array([img])])[0]
    return arr2image(np.array(r[0]))

#%%
def res_block(x, dim):
    
    x1 = Conv2D(dim, 3, padding="same", use_bias = False)(x)
    x1 = BatchNormalization()(x1, training = 1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(dim, 3, padding="same", use_bias = False)(x1)
    x1 = BatchNormalization()(x1, training = 1)
    x1 = Activation("relu")(Add()([x, x1]))
    
    return x1

def NET_G(ngf = 64, block_n = 6, downsampling_n = 2, upsampling_n = 2, image_size = 256):
    """
    Generator network 구성
    이미지 사이즈 128 => block_n = 6, 256 => block_n = 9
    
    첫번째 계층 : 크기가 7인 컨볼루션 커널 채널 
    다운 샘플링 : 크기 3의 컨볼루션 커널 크기는 레이어당 채널수의 곱셈

    """
    
    input_t = Input(shape=(image_size, image_size, 3))
    x = input_t
    dim = ngf
    
    x = Conv2D(dim, 7, padding="same")(x)
    x = Activation("relu")(x)
        
    for i in range(downsampling_n):
        dim *= 2
        x = Conv2D(dim, 3, strides = 2, padding="same", use_bias = False)(x)
        x = BatchNormalization()(x, training = 1)
        x = Activation('relu')(x)

    for i in range(block_n):
        x = res_block(x, dim)

    for i in range(upsampling_n):
        dim = dim // 2
        x = Conv2DTranspose(dim, 3, strides = 2, padding="same", use_bias = False)(x)
        x = BatchNormalization()(x, training = 1)
        x = Activation('relu')(x) 
    
    dim = 3
    x = Conv2D(dim, 7, padding="same")(x)
    x = Activation("tanh")(x)
    
    return multi_gpu_model(Model(inputs = input_t, outputs = x), gpus = None)


def NET_D(ndf = 64, max_layers = 3, image_size = 256):
    """Discriminator 구성"""
    
    input_t = Input(shape=(image_size, image_size, 3))
    x = input_t
    x = Conv2D(ndf, 4, padding = "same", strides = 2)(x)
    
    x = LeakyReLU(alpha = 0.2)(x)
    dim = ndf
    
    for i in range(1, max_layers):
        dim *= 2
        x = Conv2D(dim, 4, use_bias = False, padding = "same", strides = 2)(x)
        x = BatchNormalization()(x, training = 1)
        x = LeakyReLU(alpha = 0.2)(x)

    x = Conv2D(dim, 4, padding="same", use_bias = False)(x)
    x = BatchNormalization()(x, training = 1)
    x = LeakyReLU(alpha = 0.2)(x)
    x = Conv2D(1, 4, padding = "same", activation = "sigmoid")(x)
    
    return multi_gpu_model(Model(inputs=input_t, outputs=x), gpus=None)


def loss_func(output, target):

    return K.mean(K.square(output-target))


class CycleGAN(object):
    def __init__(self, image_size = 256, lambda_cyc = 10, lrD = 2e-4, lrG = 2e-4, ndf = 64, ngf = 64, resnet_blocks = 9):
        
        """
                      cyc loss
         +---------------------------------+      
         |            (CycleA)             |       
         v                                 |
        realA -> [GB] -> fakeB -> [GA] -> recA          
         |                 |
         |                 +---------------+
         |                                 |
         v                                 v
        [DA]         <CycleGAN>           [DB]
         ^                                 ^
         |                                 |
         +----------------+                |
                          |                |
        recB <- [GB] <- fakeA <- [GA] <- realB          
         |                                 ^
         |            (CycleB)             |
         +---------------------------------+
                        cyc loss
        """

        self.GA = NET_G(image_size = image_size, ngf = ngf, block_n = resnet_blocks)
        self.GB = NET_G(image_size = image_size, ngf = ngf, block_n = resnet_blocks)
        
        self.DA = NET_D(image_size = image_size, ndf = ndf)
        self.DB = NET_D(image_size = image_size, ndf = ndf)

        realA, realB = self.GB.inputs[0],  self.GA.inputs[0]
        
        fakeB, fakeA = self.GB.outputs[0], self.GA.outputs[0]
        recA,  recB  = self.GA([fakeB]),   self.GB([fakeA])

        self.cycleA = K.function([realA], [fakeB, recA])
        self.cycleB = K.function([realB], [fakeA, recB])

        DrealA, DrealB = self.DA([realA]), self.DB([realB])
        DfakeA, DfakeB = self.DA([fakeA]), self.DB([fakeB])

        lossDA, lossGA, lossCycA = self.get_loss(DrealA, DfakeA, realA, recA)
        lossDB, lossGB, lossCycB = self.get_loss(DrealB, DfakeB, realB, recB)

        lossG = lossGA + lossGB + lambda_cyc * (lossCycA + lossCycB)
        lossD = lossDA + lossDB

        updaterG = Adam(lr = lrG, beta_1=0.5).get_updates(lossG, self.GA.trainable_weights + self.GB.trainable_weights)
        updaterD = Adam(lr = lrD, beta_1=0.5).get_updates(lossD, self.DA.trainable_weights + self.DB.trainable_weights)
        
        self.trainG = K.function([realA, realB], [lossGA, lossGB, lossCycA, lossCycB], updaterG)
        self.trainD = K.function([realA, realB], [lossDA, lossDB], updaterD)
    
    
    def get_loss(self, Dreal, Dfake, real , rec):
        
        lossD = loss_func(Dreal, K.ones_like(Dreal)) + loss_func(Dfake, K.zeros_like(Dfake))
        lossG = loss_func(Dfake, K.ones_like(Dfake))
        lossCyc = K.mean(K.abs(real - rec))
        
        return lossD, lossG, lossCyc

    def train(self, A, B):
        
        errDA, errDB = self.trainD([A, B])
        errGA, errGB, errCycA, errCycB = self.trainG([A, B])
       
        return errDA, errDB, errGA, errGB, errCycA, errCycB
   
#%%    

IMG_SIZE = 100

DATASET = "horse2zebra"
#dataset_path = "D:\\{0}\\{0}\\".format(DATASET)
dataset_path = "E:\\Dataset\\ct-or-mri\\cnn\\test\\"
trainA_path = dataset_path + "mr(crop)\\*.png"
trainB_path = dataset_path + "ct(crop)\\*.png"

#trainA_path = dataset_path + "trainA(128)\\*.png"
#trainB_path = dataset_path + "trainB(128)\\*.png"

train_A = DataSet(trainA_path, image_size = IMG_SIZE)
train_B = DataSet(trainB_path, image_size = IMG_SIZE)

#%%
def train_batch(batchsize):
   
    epa, a = train_A.get_batch(batchsize)
    epb, b = train_B.get_batch(batchsize)
    
    return max(epa, epb), a, b

#%%
def gen(generator, X):
    # X의 그림을 생성자로 전송
    r = np.array([generator([np.array([x])]) for x in X])
    g = r[:, 0, 0]
    rec = r[:, 1, 0]
    return g, rec 

def snapshot(cycleA, cycleB, A, B):        
    """
    A、B 두 이미지의 배치
    cycleA : A->B->A
    cycleB : B->A->B
    
    사진출력:
    +-----------+     +-----------+
    | X (in A)  | ... |  Y (in B) | ...
    +-----------+     +-----------+
    |   GB(X)   | ... |   GA(Y)   | ...
    +-----------+     +-----------+
    | GA(GB(X)) | ... | GB(GA(Y)) | ...
    +-----------+     +-----------+
    """
    
    gA, recA = gen(cycleA, A)
    gB, recB = gen(cycleB, B)

    lines = [
        # np.concatenate 
        np.concatenate(A.tolist()+B.tolist(), axis = 1),
        np.concatenate(gA.tolist()+gB.tolist(), axis = 1),
        np.concatenate(recA.tolist()+recB.tolist(), axis = 1)
    ]

    arr = np.concatenate(lines)
    
   
    
    return arr2image(arr)

#%%
model = CycleGAN(image_size = IMG_SIZE)
model.GA.summary()
model.GB.summary()
model.DA.summary()
model.DB.summary()

#model = multi_gpu_model(model, gpus=None)

from IPython.display import display

#%%

save_path = 'E:\\Dataset\\ct-or-mri\\cnn\\test\\result\\'
save_img_path = save_path + 'img\\'

if not(os.path.exists(save_path)):
    os.mkdir(save_path)


EPOCH_NUM = 1000
epoch = 0

DISPLAY_INTERVAL = 200
SNAPSHOT_INTERVAL = 200

BATCH_SIZE = 2

iter_cnt = 0
err_sum = np.zeros(6)

while epoch < EPOCH_NUM:       
    
    epoch, A, B = train_batch(BATCH_SIZE) 
    err  = model.train(A, B)
    
    err_sum += np.array(err)
    iter_cnt += 1

    if (iter_cnt % DISPLAY_INTERVAL == 0):
        
        err_avg = err_sum / DISPLAY_INTERVAL
        print('[EPOCH%d] 판별손실: A %f B %f 생성손실: A %f B %f 사이클 손실: A %f B %f'
        % (iter_cnt, 
            err_avg[0], err_avg[1], err_avg[2], err_avg[3], err_avg[4], err_avg[5]),
        )   
       
        err_sum = np.zeros_like(err_sum)


    if (iter_cnt % SNAPSHOT_INTERVAL == 0):
        
        A = train_A.get_pics(4)
        B = train_B.get_pics(4)
        result_img = snapshot(model.cycleA, model.cycleB, A, B)
        display(result_img)
        scipy.misc.imsave(save_img_path + str(iter_cnt) + '.png', result_img)
        #scipy.misc.imsave(save_img_path + str(iter_cnt) + '_gA.png', gA)
        #scipy.misc.imsave(save_img_path + str(iter_cnt) + '_recA.png', recA)
        #scipy.misc.imsave(save_img_path + str(iter_cnt) + '_B.png', B)
        #scipy.misc.imsave(save_img_path + str(iter_cnt) + '_gA.png', gB)
        #scipy.misc.imsave(save_img_path + str(iter_cnt) + '_recA.png', recB)
        
        
        model.GA.save_weights(save_path + 'GA_' +str(iter_cnt) + '.h5')
        model.GB.save_weights(save_path + 'GB_' +str(iter_cnt) + '.h5')
        model.DA.save_weights(save_path + 'DA_' +str(iter_cnt) + '.h5')
        model.DB.save_weights(save_path + 'DB_' +str(iter_cnt) + '.h5')
        
        







netG_A.save_weights(save_name.format('tf_GA_weights'))
netG_B.save_weights(save_name.format('tf_GB_weights'))



load_name = dpath + '{}' + '1000.h5'
netG_A.load_weights(load_name.format('tf_GA_weights'))
netG_B.load_weights(load_name.format('tf_GB_weights'))
netD_A.load_weights(load_name.format('tf_DA_weights'))
netD_B.load_weights(load_name.format('tf_DB_weights'))
netG_train_function.load_weights(load_name.format('tf_G_train_weights'))
netD_A_train_function.load_weights(load_name.format('tf_D_A_train_weights'))
netD_B_train_function.load_weights(load_name.format('tf_D_B_train_weights'))















