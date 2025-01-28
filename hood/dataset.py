import os, random

class Dataset:

    def __init__(self, dataset_path):
        
        self.data = list()
        
        if not dataset_path: return
        
        for label in os.listdir(dataset_path):
            
            target_dir = dataset_path + str(label) + "/"
            
            for sample in os.listdir(target_dir):
                
                self.data.append([target_dir + sample, label])
            #
        #
        
        random.shuffle(self.data)
	#
#