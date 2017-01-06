# for division
from __future__ import division

# for unpacking and retreiving data
import mnist

# for classifers
import sklearn.neighbors
from sklearn import tree
# from sklearn.neural_network import MLPClassifier
import sklearn.metrics
import sklearn.ensemble
import sknn.mlp
import cPickle

#for processing image
import scipy
import skimage.morphology
import skimage.feature
from PIL import Image
from PIL import ImageFilter

# for visualizing data
from matplotlib import pyplot
import matplotlib as mpl

# for array functions
import numpy

# for mathematical functions
from math import pow
from math import sqrt

GREY_THRESHOLD = 250

# input is a PIL.Image
# returns sharpen image
def sharpen_image(image):
    # using unsharp masking technique
    return image.filter(ImageFilter.UnsharpMask())

# converts an image to black and white 
# input is a numpy array
# returns black and white image
def convert_to_bw(image):
    pil_image = Image.fromarray(image)
    pil_image = sharpen_image(pil_image)
    numpy_image =  numpy.asarray(pil_image)
    numpy_image.flags.writeable = True
    numpy_image[numpy_image < GREY_THRESHOLD]=0
    numpy_image[numpy_image >= GREY_THRESHOLD]=255
    return numpy_image

# input is a 28 x 28 image array
# divides the image into four zones
# and calculates the ratio between
# black and total pixels in each zones
# returns ratio between black_pixels and
# total pixels in each zone
def get_distribution_sequence(image):
    x1 = 0
    x2 = 14
    y1 = 0
    y2 = 14
    zone1_black_pixels = 0
    zone2_black_pixels = 0
    zone3_black_pixels = 0
    zone4_black_pixels = 0
        
    while x1 < 14:
        y1 = 0
        y2 = 14
        while y1 < 14:
            if image[x1][y1] >= GREY_THRESHOLD:
                zone1_black_pixels += 1
            
            if image[x1][y2] >= GREY_THRESHOLD:
                zone2_black_pixels += 1
                
            if image[x2][y1] >= GREY_THRESHOLD:
                zone3_black_pixels += 1
                
            if image[x2][y2] >= GREY_THRESHOLD:
                zone4_black_pixels += 1
                
            y1 += 1
            y2 += 1
            
        x1 += 1
        x2 += 1
    
    total_pixels = 14 * 14
    
    distribution1 = round(zone1_black_pixels/total_pixels,2)
    distribution2 = round(zone2_black_pixels/total_pixels,2)
    distribution3 = round(zone3_black_pixels/total_pixels,2)
    distribution4 = round(zone4_black_pixels/total_pixels,2)
    
    return [distribution1, distribution2, distribution3, distribution4]
    
# input is an image array
# returns ratio between pixels loops and total area
def get_loop_ratio(image):
    numpy_image = convert_to_bw(image)
    numpy_image = skimage.morphology.skeletonize(numpy_image>128)
    filled_holes = scipy.ndimage.binary_fill_holes(numpy_image)
    x = 0
    y = 0
    row = filled_holes.shape[0]
    col = filled_holes.shape[1]
    while x < row:
        y = 0
        while y < col:
            count = 0
            if filled_holes[x][y] == True:
                if(x != 0 and filled_holes[x-1][y] != False):
                    count += 1
                    
                if(x != row -1 and filled_holes[x+1][y] != False):
                    count += 1
                    
                if(y != 0 and filled_holes[x][y-1] != False):
                    count += 1
                    
                if(y != col - 1 and filled_holes[x][y+1] != False):
                    count += 1
                
                
                if(x != 0 and y != 0 and filled_holes[x-1][y-1] != False):
                    count += 1
                
                if(x != 0 and y != col -1 and filled_holes[x-1][y+1] != False):
                    count += 1
                    
                if(x != row -1 and y != 0 and filled_holes[x+1][y-1] != False):
                    count += 1
                
                if(x != row - 1 and y != col - 1 and filled_holes[x+1][y+1] != False):
                    count += 1
                
                if (count <= 2):
                    filled_holes[x][y] = False
                
            y += 1
            
        x += 1

    total_pixels = row * col
    black_pixels = 0
    x = 0
    y = 0
    while x < row:
        y = 0
        while y < col:
            if(filled_holes[x][y] == True):
                black_pixels += 1
                
            y += 1
        x += 1

    return round(black_pixels/total_pixels,2)

def get_loop_count(image):
    numpy_image = convert_to_bw(image)
    numpy_image = skimage.morphology.skeletonize(numpy_image>128)
    filled_holes = scipy.ndimage.binary_fill_holes(numpy_image)
    loop_count = 0
    if( ~(numpy_image == filled_holes).all() == True):
        x = 0
        y = 0
        row = filled_holes.shape[0]
        col = filled_holes.shape[1]
        
        while(x < row):
            y = 0
            while(y < col):
                count = 0
                
                if(filled_holes[x][y] != False):
                    if(x != 0 and filled_holes[x-1][y] != False):
                        count += 1
                        
                    if(x != row -1 and filled_holes[x+1][y] != False):
                        count += 1
                        
                    if(y != 0 and filled_holes[x][y-1] != False):
                        count += 1
                        
                    if(y != col - 1 and filled_holes[x][y+1] != False):
                        count += 1
                        
                    if(x != 0 and y != 0 and filled_holes[x-1][y-1] != False):
                        count += 1
                    
                    if(x != 0 and y != col -1 and filled_holes[x-1][y+1] != False):
                        count += 1
                        
                    if(x != row -1 and y != 0 and filled_holes[x+1][y-1] != False):
                        count += 1
                    
                    if(x != row - 1 and y != col - 1 and filled_holes[x+1][y+1] != False):
                        count += 1
                
                if (count <= 2):
                    filled_holes[x][y] = False
                    #print (x, y, end_point_count)
                    
                y += 1
            
            x += 1
        
        
        labeled_array, loop_count = scipy.ndimage.label(filled_holes)
        
    return loop_count

def convex_hull(image):
    numpy_image = convert_to_bw(image)
    numpy_image = skimage.morphology.skeletonize(numpy_image>128)
    #numpy_image = ~numpy_image
    bool_image = skimage.morphology.convex_hull_image(numpy_image)
    bool_image = skimage.morphology.skeletonize(bool_image>128)
    show_image(bool_image)
    
    
def gradient_descent(option = 1):
    train_data = numpy.genfromtxt("train_features.csv",delimiter=',')
    X1 = train_data[:,1:train_data.shape[1]]
    y1 = train_data[:,0]
    print 'Training classifier'
    activation_function = ''
    if option == 1:
        activation_function = 'identity'
    
    elif option == 2:
        activation_function = 'sigmoid'
    
    elif option == 3:
        activation_function = 'squash'
        
    if activation_function == 'sigmoid':
        clf = sklearn.linear_model.SGDClassifier('log', learning_rate ='optimal', n_jobs = 1)
   
    elif activation_function == 'identity':
        clf = sklearn.linear_model.SGDClassifier('perceptron',learn_rate = 'optimal', n_jobs = 1)
        
    elif activation_function == 'squash':
        clf = sklearn.linear_model.SGDClassifier(learn_rate = 'optimal', n_jobs = 1)
        
    clf.fit(X=X1,y=y1)    
        
    print 'Testing classifier'
    test_data = numpy.genfromtxt("test_features.csv", delimiter=',')
    X2 = test_data[:,1:test_data.shape[1]]
    y2 = test_data[:,0]

    return clf.score(X2, y2)
    
# input is a numpy array
# returns number of end points of the image
def get_end_points_count(image):    
    numpy_image = convert_to_bw(image)
    numpy_image = skimage.morphology.skeletonize(numpy_image>128)
    
    row = numpy_image.shape[0]
    col = numpy_image.shape[1]
    
    x = 0
    y = 0
    count = 0
    end_point_count = 0
    
    while(x < row):
        y = 0
        while(y < col):
            count = 0
            
            if(numpy_image[x][y] != False):
                if(x != 0 and numpy_image[x-1][y] != False):
                    count += 1
                    
                if(x != row -1 and numpy_image[x+1][y] != False):
                    count += 1
                    
                if(y != 0 and numpy_image[x][y-1] != False):
                    count += 1
                    
                if(y != col - 1 and numpy_image[x][y+1] != False):
                    count += 1
                    
                if(x != 0 and y != 0 and numpy_image[x-1][y-1] != False):
                    count += 1
                
                if(x != 0 and y != col -1 and numpy_image[x-1][y+1] != False):
                    count += 1
                    
                if(x != row -1 and y != 0 and numpy_image[x+1][y-1] != False):
                    count += 1
                
                if(x != row - 1 and y != col - 1 and numpy_image[x+1][y+1] != False):
                    count += 1
            
            if (count == 1):
                end_point_count += 1
                #print (x, y, end_point_count)
                
            y += 1
            
        x += 1
                
    return end_point_count
    
# input is image array
# returns average distances of zones to image centroid
def distance_image_centroid(image):
    centroidX = image.shape[0]/2
    centroidY = image.shape[1]/2
    x1 = 0
    y1 = 0
    x2 = 14
    y2 = 14

    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    
    while x1 < 14:
        y1 = 0
        y2 = 14
        while y1 < 14:
            
            if image[x1][y1] >= GREY_THRESHOLD:
                sum1 += sqrt(pow(centroidX-x1, 2) + pow(centroidY-y1,2))
                n1 += 1
            
            if image[x1][y2] >= GREY_THRESHOLD:
                sum2 += sqrt(pow(centroidX-x1, 2) + pow(centroidY-y2,2))
                n2 += 1
                
            if image[x2][y1] >= GREY_THRESHOLD:
                sum3 += sqrt(pow(centroidX-x2, 2) + pow(centroidY-y1,2))
                n3 += 1
                
            if image[x2][y2] >= GREY_THRESHOLD:
                sum4 += sqrt(pow(centroidX-x2, 2) + pow(centroidY-y2,2))
                n4 += 1
                
            y1 += 1
            y2 += 1
            
        x1 += 1
        x2 += 1
    
    if n1>0:
        avg1 = round(sum1/n1,2)        
    else:
        avg1 = 0

    if n2>0:       
        avg2 = round(sum2/n2,2)
    else:
        avg2 = 0
    
    if n3>0:
        avg3 = round(sum3/n3,2)
    else:
        avg3 = 0
        
    if n4>0:
        avg4 = round(sum4/n4,2)
    else:
        avg4 = 0
        
    return [avg1, avg2, avg3, avg4]

def hog(image):
    return skimage.feature.hog(image)    
    
def show_image(image):
    pyplot.imshow(image,cmap=mpl.cm.Greys,interpolation='none')
    pyplot.axis("off")

def save_image(image, file_name):
    pyplot.imshow(image,cmap=mpl.cm.Greys,interpolation='none')
    pyplot.axis("off")
    pyplot.savefig("img3", bborder = "tight",transparent=True )


def main1():
    train_images, train_labels = mnist.get_labeled_data("training")
    #test_images, test_labels = mnist.get_labeled_data("testing")
    picture_number = 7
    image = train_images[picture_number]
    #end_points = get_end_points_count(train_images[picture_number])     
    a = hog(image)
    a = a
    #image = skimage.morphology.skeletonize(train_images[picture_number]>128)

def generate_pixels():
    train_images, train_labels = mnist.get_labeled_data("training")
    train_labels = train_labels.reshape(train_labels.shape[0],)
    i = 0
    size = train_images.shape[0]
    while i < size:
        image = train_images[i,:,:]
        image = image.reshape(784,)
        return
    
def generate_features():
    train_images, train_labels = mnist.get_labeled_data("training")
    train_labels = train_labels.reshape(train_labels.shape[0],)
    i = 0
    size = train_images.shape[0]
    features = numpy.zeros((size,90),dtype=numpy.float32)
    
    j = 0
    #size += 1
    print "Extracting features"
    #size = 5
    while i < size:
        features[i][0]=train_labels[j]
        features[i][1]=get_loop_count(train_images[j])
        features[i][2]=get_end_points_count(train_images[j])
        features[i][3], features[i][4], features[i][5], features[i][6] = get_distribution_sequence(train_images[j])
        features[i][7] =get_loop_ratio(train_images[j])
        features[i][8:89]=hog(train_images[j])
        j += 1
        i += 1
        
    numpy.savetxt("train_features.csv", features, delimiter=',')

    test_images, test_labels = mnist.get_labeled_data("testing")
    i = 0
    size = test_images.shape[0]
    test_labels = test_labels.reshape(test_labels.shape[0],)
    test_features = numpy.zeros((size,90),dtype=numpy.float32)

    j = 0
    #size += 1
    print "Extracting features"
    while i < size:
        print "image " + str(i)
        test_features[i][0]=test_labels[i]
        test_features[i][1]=get_loop_count(test_images[i])
        test_features[i][2]=get_end_points_count(test_images[i])
        test_features[i][3], test_features[i][4], test_features[i][5], test_features[i][6] = get_distribution_sequence(test_images[j])
        test_features[i][7] =get_loop_ratio(test_images[j])
        test_features[i][8:89]=hog(test_images[j])
        j += 1        
        i += 1

    numpy.savetxt("test_features.csv", test_features, delimiter=',')

def knn():
    train_data = numpy.genfromtxt("train_features.csv",delimiter=',')
    X1 = train_data[:,1:train_data.shape[1]]
    y1 = train_data[:,0]
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)
    print 'Training classifier'
    neigh.fit(X1, y1)
    
    with open('k-nn', 'wb') as fid:
        cPickle.dump(neigh, fid)    
    
    print 'Testing classifier'
    test_data = numpy.genfromtxt("test_features.csv", delimiter=',')
    X2 = test_data[:,1:test_data.shape[1]]
    y2 = test_data[:,0]
    return neigh.score(X2, y2)

def decision_tree(depth = 10):
    train_data = numpy.genfromtxt("train_features.csv",delimiter=',')
    X1 = train_data[:,1:train_data.shape[1]]
    y1 = train_data[:,0]
    print 'Training classifier'
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(X1, y1)
    
    idArray = numpy.empty((train_data.shape[1],1), dtype='S20')
    idArray[0] = 'loop_count'
    idArray[1] = 'end_points'
    idArray[2] = 'distribution_sequence_1'
    idArray[3] = 'distribution_sequence_2'
    idArray[4] = 'distribution_sequence_3'
    idArray[5] = 'distribution_sequence_4'
    idArray[6] = 'loop_ratio'
    idArray[7:88] = 'hog'

    with open('graph.dot', 'w') as file:
        tree.export_graphviz(clf, out_file = file, feature_names = idArray)

    file.close()
    
    print 'Testing classifier'
    test_data = numpy.genfromtxt("test_features.csv", delimiter=',')
    X2 = test_data[:,1:test_data.shape[1]]
    y2 = test_data[:,0]

    print clf.score(X2, y2)
    
def neural_network():
    train_data = numpy.genfromtxt("train_features.csv",delimiter=',')
    X1 = train_data[:,1:train_data.shape[1]]
    y1 = train_data[:,0]
    print 'Training classifier'
    layer1 = sknn.mlp.Layer('Sigmoid', units = 15)
    layer3 = sknn.mlp.Layer('Softmax', units = 10)
    layersList = [layer1, layer3]
    
    clf = sknn.mlp.Classifier(layersList, verbose=True, n_iter=25)
    clf.fit(X=X1,y=y1)
    
    print 'Testing classifier'
    test_data = numpy.genfromtxt("test_features.csv", delimiter=',')
    X2 = test_data[:,1:test_data.shape[1]]
    y2 = test_data[:,0]
    
    with open('nn_classifier.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)    
    
    
    print clf.score(X=X2, y=y2)
    
    ''''
    
    clf = MLPClassifier(activation='logistic', algorithm='sgd')
    clf.fit(X1, y1)
    
    
    '''
    return 0
    
def ada_boost():
    train_data = numpy.genfromtxt("train_features.csv",delimiter=',')
    X1 = train_data[:,1:train_data.shape[1]]
    y1 = train_data[:,0]
    print 'Training classifier'
    clf = sklearn.ensemble.AdaBoostClassifier()
    clf.fit(X1, y1)
    
    with open('ada_boost.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)
    
    print 'Testing classifier'
    test_data = numpy.genfromtxt("test_features.csv", delimiter=',')
    X2 = test_data[:,1:test_data.shape[1]]
    y2 = test_data[:,0]

    print clf.score(X2, y2)


if __name__ == '__main__':
    generate_pixels()