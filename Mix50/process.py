import soundfile as sf
from IPython.display import Audio
import numpy as np

class Process:
    '''
    This class supports the effects module as a wrapper for further manipulations.
    '''
    
    def __init__(self, audio):
        
        '''
        Set the audio from our mixfifty module
        '''
           
        self.y, self.sr = audio, 22050
       
        
    def play(self):
        '''
        Plays the audio using IPython's Audio display.

        '''
        return Audio(self.y, rate=22050)
    
    def save(self, file_path: str):
        '''
        Saves the audio data to a file in PCM 24-bit format. MP3 is not supported! Please use .WAV!
        '''
        sf.write(file_path, self.y, 22050, 'PCM_24')
        return f"Exported to {file_path}"
    
    def raw_audio(self) -> np.ndarray:
        '''
        Return the raw audio as an nd.array
        '''
        return self.y
