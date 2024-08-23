# Mix50/mixfifty.py
from .effects import Effects
from .features import Features
from .process import Process
import librosa


class MixFifty:
    
    def __init__(self):
        
        #Initialize all modules in 
        self.effects = Effects()
        self.process = Process()
        self.features = Features()


    def load_audio(self, path1, path2=None):
        
        self.path1 = path1
        self.path2 = path2
        self.y1, self.sr1 = librosa.load(path1, sr=22050)
        
        if path2:
            self.y2, self.sr2 = librosa.load(path2, sr=22050)
        else:
            self.y2 = None
            self.sr2 = None

        # Pass the loaded audio to classes
        self.effects.set_audio(self.y1, self.sr1, self.y2, self.sr2)
        self.features.set_audio(self.y1, self.sr1, self.y2, self.sr2)
        self.process.set_audio(self.y1, self.sr1, self.y2, self.sr2)
