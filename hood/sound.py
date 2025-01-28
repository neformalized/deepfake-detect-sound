from pydub import AudioSegment

class Encoder:
    
    def __init__(self, duration, sample_rate):
        
        self.duration = duration
        self.sample_rate = sample_rate
    #
    
    def process(self, sound_path):
        
        sample = AudioSegment.from_wav(sound_path)
        
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        image = image/255
        
        return image
    #
#