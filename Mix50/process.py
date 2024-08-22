import soundfile as sf
from IPython.display import Audio


class process:
    '''
    This class supports the effects module as a wrapper for further manipulations

    '''
    
    def __init__(self, audio):
        self.audio = audio
    
    def play(self):
        return Audio(self.audio, rate=22050)
    
    def save(self, file_path: str):
        sf.write(file_path, self.audio, 22050, 'PCM_24')
        return f"Exported to {file_path}"