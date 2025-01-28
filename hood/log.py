class Log:

    def __init__(self, save_path, headers):
        
        self.log_path = save_path + "log.txt"
        
        #
        
        self.title(headers)
    #
    
    def title(self, headers):
        
        with open(self.log_path, "w", encoding="utf8") as file:
            
            for key, value in headers.items():
                file.write(f"{key}: {value}\n")
            #
            
            file.write("---------------------\n")
            file.write("-=-=-=-=-=-=-=-=-=-=-\n")
        #
    #
    
    def line(self, data):
        
        with open(self.log_path, "a", encoding="utf8") as file:
            
            for key, value in data.items():
                file.write(f"{key}:{value}\n")
            #
            
            file.write("-=-=-=-=-=-=-=-=-=-=-\n")
        #
    #
#