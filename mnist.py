import os
import struct
import numpy as np
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import matplotlib as mpl
from matplotlib import pyplot

def get_labeled_data(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def unpack_miniset(file_name):
    i=18672
    while(i<60000):
        print i        
        image = read_image(file_name,i)
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.axis("off")
        pyplot.savefig("images/"+str(i),bborder="tight",transparent="true")
        pyplot.close()
        i+=1

def read_image(file_name, idx_image):
	"""
		file_name: If used for the MNIST dataset, should be either 
					train-images-idx3-ubyte or t10k-images-idx3-ubyte
		idx_image: index of the image you want to read.
	"""
	img_file = open(file_name,'r+b')
	img_file.seek(0)
	magic_number = img_file.read(4)
	magic_number = struct.unpack('>i',magic_number)
 
	data_type = img_file.read(4)
	data_type = struct.unpack('>i',data_type)

	dim = img_file.read(8)
	dimr = struct.unpack('>i',dim[0:4])
	dimr = dimr[0]
	dimc = struct.unpack('>i',dim[4:])
	dimc = dimc[0]

	image = np.ndarray(shape=(dimr,dimc))
	img_file.seek(16+dimc*dimr*idx_image)
	
	for row in range(dimr):
		for col in range(dimc):
			tmp_d = img_file.read(1)
			tmp_d = struct.unpack('>B',tmp_d)
			image[row,col] = tmp_d[0]
	
	img_file.close()
	return image