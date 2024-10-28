#!/usr/bin/env/ python
# ECBM E4040 Fall 2024 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
import os

try:
    from scipy.ndimage.interpolation import rotate
    from scipy.ndimage import zoom, gaussian_filter
except ModuleNotFoundError:
    os.system('pip install scipy')
    from scipy.ndimage.interpolation import rotate
    from scipy.ndimage import zoom, gaussian_filter

class ImageGenerator(object):
    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.

        Inputs:
            :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
            :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """
        #######################################################################
        # TODO: Your ImageGenerator instance must store the following:        #
        #           x, y, num_of_samples, height, width, number of pixels,    #
        #           translated, degree of rotation, is_horizontal_flip,       #
        #           is_vertical_flip, is_add_noise.                           #
        #       By default, set boolean values to False.                      #
        #                                                                     #
        # Hint: Since you may directly perform transformations on x and y,    #
        #       and don't want your original data to be contaminated by       #
        #       those transformations, you should use numpy array's           #
        #       build-in copy() method.                                       #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        raise NotImplementedError
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################

        # One way to use augmented data is to store them after transformation
        # (and then combine all of them to form a new dataset)
        # The following variables (along with create_aug_data() function) is one
        # way of implementing this. You can either figure out how to use them or
        # find out your own ways to create the augmented dataset.
        
        # If you have your own idea of creating an augmented dataset, feel free
        # to comment out any code you don't need.
        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.bright = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N
    
    
    def create_aug_data(self):
        # If you want to use the function create_aug_data() to generate a new dataset, you can perform the following operations in each
        # transformation function:
        #
        # 1. Store the transformed data with their labels in a tuple called self.translated, self.rotated, self.flipped, etc. 
        # 2. Increase self.N_aug by the number of transformed data,
        # 3. You should also return the transformed data in order to show them in the task4 notebook
        
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
        if self.bright:
            self.x_aug = np.vstack((self.x_aug,self.bright[0]))
            self.y_aug = np.hstack((self.y_aug,self.bright[1]))
            
        print("Size of training data:{}".format(self.N_aug))
        
    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        Inputs:
            :param batch_size: The number of samples to return for each batch.
            :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                            If False, the order or data samples stays the same.
        
        :return: 
            A batch of data with size (batch_size, width, height, channels).
        """

        #######################################################################
        # TODO: Use the 'yield' keyword to implement this generator.          #
        #       Pay attention to the following:                               #
        #       1. The generator should return batches endlessly.             #
        #       2. Make sure the shuffle only happens after each sample has   #
        #          been visited once. Else some samples might not appear.     #
        #                                                                     #
        #---------------------------------------------------------------------#
        # One possible pseudo code for your reference:                        #
        #---------------------------------------------------------------------#
        #   calculate the total number of batches possible                    #
        #   (if the rest is not sufficient to make up a batch, ignore)        #
        #   while True:                                                       #
        #       if (batch_count < total number of batches possible):          #
        #           batch_count = batch_count + 1                             #
        #           yield(next batch of x and y indicated by batch_count)     #
        #       else:                                                         #
        #           shuffle(x)                                                #
        #           reset batch_count                                         #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        raise NotImplementedError
    
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        Inputs:
            :param images: images to be shown
        """
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        raise NotImplementedError
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        
        Inputs:
            :param angle: Rotation angle in degrees.
        
        :return: 
            rotated dataset
        """         
        self.dor = angle
        rotated = rotate(self.x.copy(), angle,reshape=False,axes=(1, 2))
        print('Currrent rotation: ', self.dor)
        self.rotated = (rotated, self.y.copy())
        self.N_aug += self.N
        return rotated
    
    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        
        Inputs:
            :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                            then 1000 samples will be noise-injected.
            :param amplitude: An integer scaling factor of the noise.
        
        :return: 
            dataset with noise added
        """

        assert portion <= 1
        if not self.is_add_noise:
            self.is_add_noise = True
        m = self.N * portion
        index = np.random.choice(self.N, m, replace=False)
        added = self.x.copy()
        for i in index:
            added[i, :, :, :] += np.random.randint(0, 5, [self.height, self.width, self.channel], dtype='uint8') * amplitude
        self.added = (added, self.y.copy()[index])
        self.N_aug += m
        return added
    
    def brightness(self, factor):
        """
        Scale the pixel values to increase the brightness.

        Inputs:
            :param factor: A factor (>=1) by which each pixel in the image will be scaled. 
                        For instance, if the factor is 2, all pixel values will be doubled.
        
        :return: 
            dataset with increased brightness
        """
        assert factor >= 1, "Factor should be greater than or equal to 1"
        
        bright = self.x.copy()
        
        # if not self.is_bright:
        #     self.is_bright = True
        
        # Scaling pixel values and ensuring they don't exceed 255
        bright = (bright * factor).astype(int)
        bright[bright >= 255] = 255
        
        print("Brightness increased by a factor of:", factor)
        
        self.bright = (bright, self.y.copy())
        self.N_aug += self.N
        return bright
    
    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        
        Inputs:
            :param shift_height: the number of pixels to shift along height direction. Can be negative.
            :param shift_width: the number of pixels to shift along width direction. Can be negative.
        
        :return: 
            translated dataset
        """
        #######################################################################
        # TODO: Implement the translate() function. You may wonder what       #
        #       values to append to the edge after the shift. Here, use       #
        #      rolling instead. For example, if you shift 3 pixels to the     #
        #      left, append the left-most 3 columns that are out of boundary  #
        #      to the right edge of the picture.                              #
        #                                                                     #
        # HINT: use np.roll                                                   #
        # https://numpy.org/doc/stable/reference/generated/numpy.roll.html    #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        
        raise NotImplementedError
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################
    
    
    def random_resized_crop(self,  crop_size = (28,28)):
        """
        Randomly crops a region of each image in the batch and resizes it to the original size (28x28).

        Inputs:
            :param crop_size: Desired crop size as a tuple (crop_height, crop_width).
                        Default is (28, 28) for consistency with other functions.
        :return: 
            The batch of resized cropped images.
        """
        #######################################################################
        # TODO: Implement the random_resized_crop function.                   #
        #                                                                     #
        # This function should perform the following tasks:                   #
        # 1. Randomly select a top-left corner within the image dimensions.   #
        # 2. Crop a region of size specified by `crop_size`.                  #
        # 3. Resize the cropped region back to the original size (28x28).     #
        #  If you give input as (6,7) it randomly crops a (6,7) region in     #
        # the image and resizes the (6,7) region back to original image's size#
        #                                                                     #
        # Hint: Use random.randint to select the top-left corner and scipy's  #
        # zoom (scipy.ndimage.zoom) function to resize the cropped region     #
        #                                                                     #
        # Ensure the crop size isn't greater than the image size              #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################

        raise NotImplementedError
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################

    
    def Gaussian_blur(self, sigma):
        """
        Apply Gaussian blur to a batch of grayscale images.

        Inputs:
            :param sigma: Standard deviation for Gaussian kernel.
        :return: 
            Blurred images with the same shape as the input.
        """
        #######################################################################
        # TODO: Implement Gaussian blur for a batch of grayscale images.      #
        #       particular batch of image's mean and standard deviation       #
        # HINT: use scipy.ndimage.gaussian_filter                             #
        #######################################################################

        raise NotImplementedError
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################
      
