from hood import dataset, buffer, builder, sound, label, log

class Handler():
    
    def __init__(self, dataset_train_path, dataset_validation_path, buffer_size, segment_duration, segment_sr, output_labels, metric_target, save_path):
        
        self.dataset_train = dataset.Dataset(dataset_train_path)
        self.dataset_validation = dataset.Dataset(dataset_validation_path)
        
        #
        
        self.buffer_size = buffer_size
        
        #
        
        self.segment_duration = segment_duration
        self.segment_sr = segment_sr
        
        #
        
        self.output_labels = output_labels
        
        #
        
        self.encoder_input = sound.Encoder(self.segment_duration, self.segment_sr)
        self.encoder_output = label.Encoder()
        
        #
        
        self.model = builder.create(int(self.segment_duration * self.segment_sr), self.output_labels)
        
        #
        
        self.metric_target = metric_target
        
        #
        
        self.save_path = save_path
        
        #
        
        log_headers = {
            "train": len(self.dataset_train.data),
            "validation": len(self.dataset_validation.data)
        }
        
        self.log = log.Log(self.save_path, log_headers)
        
        #
        
        self.epoch = 1
        
        #
        
        self.loss_train_current = 1.
        self.loss_train_last = 1.
        self.loss_train_record = 1.
        
        #
        
        self.confusion_matrix_current = list()
        
        #
        
        self.accuracy_validation_current = 0.
        self.accuracy_validation_last = 0.
        self.accuracy_validation_record = 0.
        
        #
        
        self.recall_validation_current = 0.
        self.recall_validation_last = 0.
        self.recall_validation_record = 0.        
        
        #
        
        self.precision_validation_current = 0.
        self.precision_validation_last = 0.
        self.precision_validation_record = 0.
        
        #
        
        self.specificity_validation_current = 0.
        self.specificity_validation_last = 0.
        self.specificity_validation_record = 0.
        
        #
        
        self.f1_validation_current = 0.
        self.f1_validation_last = 0.
        self.f1_validation_record = 0.
        
        #
        
        self.buffer = buffer.Dual(self.buffer_size, int(self.segment_duration * self.segment_sr), self.output_labels)
    #
    
    def start(self):
        
        while True:
            
            #
            
            self.fit()
            
            #
            
            self.evaluate()
            
            #
            
            self.checkpoint()
            
            #
            
            self.write_log()
            
            #
            
            if self.is_done(): break
            
            #
            
            self.updates()
        #
    #
    
    def fit(self):
        
        print("epoch#{}".format(self.epoch))
        
        #
        
        train_len = len(self.dataset_train.data)
        
        #
        
        losses = []
        
        for start in range(0, train_len, self.buffer_size):
            
            end = min(start + self.buffer_size, train_len)
            
            #
            
            print("step {} from {}".format([start, end], train_len))
            
            #
            
            self.buffer_load(self.dataset_train.data[start: end], self.buffer)
            
            #
            
            losses.append(self.model.fit(self.buffer.x[0: end - start], self.buffer.y[0: end - start], batch_size = 1, epochs = 1).history["loss"][0])
            
            #
        #
        
        self.loss_train_current = round(sum(losses)/len(losses), 2)
        
        #
    #
    
    def evaluate(self):
        
        if len(self.dataset_validation.data) == 0: return
        
        #
        
        print("-------------------------------")
        
        #
        
        evaluate_len = len(self.dataset_validation.data)
        
        #
        
        CM = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        
        #
        
        for start in range(0, evaluate_len, self.buffer_size):
            
            end = min(start + self.buffer_size, evaluate_len)
            
            #
            
            print("step {} from {}".format([start, end], evaluate_len))
            
            #
            
            self.buffer_load(self.dataset_validation.data[start: end], self.buffer)
            
            #
            
            for index in range(end - start):
                
                sample, real = self.buffer.get(index)
                
                #
                
                predicted = self.model.predict(sample)
                
                #
                
                loss = abs(predicted[0][0] - real[0])
                
                #
                
                if real[0] == 1.:
                    
                    if loss < 0.5: CM["TP"] += 1
                    else: CM["FP"] += 1
                else:
                    
                    if loss < 0.5: CM["TN"] += 1
                    else: CM["FN"] += 1
                #
            #
        #
        
        try:
            self.accuracy_validation_current = round((CM["TP"] + CM["TN"])/(CM["TP"] + CM["TN"] + CM["FP"] + CM["FN"]), 2)
        except:
            self.accuracy_validation_current = 0.
        #
        
        try:
            self.recall_validation_current = round(CM["TP"]/(CM["TP"] + CM["FN"]), 2)
        except:
            self.recall_validation_current = 0.
        #
        
        try:
            self.precision_validation_current = round(CM["TP"]/(CM["TP"] + CM["FP"]), 2)
        except:
            self.precision_validation_current = 0.
        #
        
        try:
            self.specificity_validation_current = round(CM["TN"]/(CM["TN"] + CM["FP"]), 2)
        except:
            self.specificity_validation_current = 0.
        #
        
        try:
            self.f1_validation_current = round(2 * ((self.precision_validation_current * self.recall_validation_current)/(self.precision_validation_current + self.recall_validation_current)), 2)
        except:
            self.f1_validation_current = 0.
        #
        
        self.confusion_matrix_current = CM
    #

    def checkpoint(self):
        
        self.model.save(self.save_path + "model.keras")
        
        if(self.accuracy_validation_current > self.accuracy_validation_record):
            
            self.model.save(self.save_path + "model_record.keras")
            
            print("record!")
        #
    #
    
    def write_log(self):
        
        log = {
            "epoch": self.epoch,
            "t. loss": self.loss_train_current,
            "v. accuracy": self.accuracy_validation_current,
            "v. recall": self.recall_validation_current,
            "v. precision": self.precision_validation_current,
            "v. specificity": self.specificity_validation_current,
            "v. f1": self.f1_validation_current,
            "v. confusion matrix": self.confusion_matrix_current
        }
        
        self.log.line(log)
    #
    
    def is_done(self):
        
        if(self.accuracy_validation_current >= self.metric_target): return True
        
        #
        
        return False
    #
    
    def updates(self):
        
        #
        # some logic to update learning rate or something
        #
        
        #
        
        if(self.loss_train_current < self.loss_train_record): self.loss_train_record = self.loss_train_current
        self.loss_train_last = self.loss_train_current
        
        #
        
        if(self.accuracy_validation_current > self.accuracy_validation_record): self.accuracy_validation_record = self.accuracy_validation_current
        self.accuracy_validation_last = self.accuracy_validation_current
        
        #
        
        if(self.recall_validation_current > self.recall_validation_record): self.recall_validation_record = self.recall_validation_current
        self.recall_validation_last = self.recall_validation_current
        
        #
        
        if(self.precision_validation_current > self.precision_validation_record): self.precision_validation_record = self.precision_validation_current
        self.precision_validation_last = self.precision_validation_current
        
        #
        
        if(self.specificity_validation_current > self.specificity_validation_record): self.specificity_validation_record = self.specificity_validation_current
        self.specificity_validation_last = self.specificity_validation_current
        
        #
        
        if(self.f1_validation_current > self.f1_validation_record): self.f1_validation_record = self.f1_validation_current
        self.f1_validation_last = self.f1_validation_current
        
        #
        
        self.epoch += 1
    #
    
    # # # # #
    
    def buffer_load(self, data, target_buffer):
        
        for i, item in enumerate(data):
            
            x = self.encoder_input.process(item[0])
            y = self.encoder_output.process(item[1])
            
            target_buffer.put(i, x, y)
        #
    #
#