from hood import handler

if __name__ == "__main__":
    
    #
    
    dataset_path_train = "/content/dataset/train/"
    dataset_path_validation = "/content/dataset/validation/"
    #dataset_path_validation = False
    
    #
    
    buffer_size = 100
    
    #
    
    segment_duration = 2
    segment_sr = 24000
    
    #
    
    output_labels = 1
    
    #
    
    
    metric_target = 0.9
    
    #
    
    save_path = "/content/"
    
    #
    
    handler = handler.Handler(dataset_path_train, dataset_path_validation, buffer_size, segment_duration, segment_sr, output_labels, metric_target, save_path)
    handler.start()
    
    #
#