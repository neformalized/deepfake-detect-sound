from pydub import AudioSegment

class Encoder:
    
    def __init__(self, duration, sample_rate):
        
        self.duration = duration
        self.sample_rate = sample_rate
    #
    
    def process(self, sound_path):
        
        sample = AudioSegment.from_file(sound_path)
        
        sample = sample.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
        
        sample = list(sample[:int(self.duration * 1000)].get_array_of_samples())
        
        sample = [impulse / 32768.0 for impulse in samples]
        
        while len(sample) < int(self.duration * self.sample_rate):
            
            sample.append(0.0)
        #
        
        return sample[:int(self.duration * self.sample_rate)]
    #
#