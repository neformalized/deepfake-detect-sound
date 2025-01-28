import numpy

class Dual:

    def __init__(self, size, shape_x, shape_y):
        
        self.x = numpy.zeros((size, shape_x, 1), dtype = numpy.float16)
        self.y = numpy.zeros((size, shape_y), dtype = numpy.float16)
    #
    
    def put(self, index, x, y):
        
        self.x[index,:,:] = numpy.expand_dims(x, axis = -1)
        self.y[index,:] = y
    #
    
    def get(self, index):
        
        return [numpy.expand_dims(self.x[index,:,:], axis = 0), self.y[index,:]]
    #
#