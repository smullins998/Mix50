# Mix50/mixfifty.py
from .effects import Effects
from .features import Features
from .process import Process
from .transitions import Transitions
import librosa


class MixFifty:
    """
    A class for handling audio processing and effects using various modules.

    Attributes
    ----------
    path1 : str
        The file path for the first audio file.
    path2 : str, optional
        The file path for the second audio file (default is None).
    y1 : ndarray
        The audio time series data for the first audio file.
    sr1 : int
        The sample rate of the first audio file.
    y2 : ndarray, optional
        The audio time series data for the second audio file (default is None).
    sr2 : int, optional
        The sample rate of the second audio file (default is None).

    Parameters
    ----------
    path1 : str
        The file path to the first audio file to be loaded.
    path2 : str, optional
        The file path to the second audio file to be loaded (default is None).
    """
    
    def __init__(self):
        """
        Initializes the MixFifty instance by creating instances of the Effects,
        Process, and Features classes.
        """
        # Initialize all modules
        self.effects = Effects()
        self.features = Features()
        self.transitions = Transitions()

    
    def load_audio(self, path1, path2=None):
        """
        Loads audio data from the specified file paths and passes it to the
        Effects, Features, and Process modules.

        Parameters
        ----------
        path1 : str
            The file path to the first audio file to be loaded.
        path2 : str, optional
            The file path to the second audio file to be loaded (default is None).

        Notes
        -----
        The audio files are loaded using librosa with a sample rate of 22050 Hz.
        If only one audio file is provided, the second audio is set to None.
        """
        self.path1 = path1
        self.path2 = path2
        self.y1, self.sr1 = librosa.load(path1, sr=22050)
        
        if path2:
            self.y2, self.sr2 = librosa.load(path2, sr=22050)
        else:
            self.y2 = None
            self.sr2 = None

        # Pass the loaded audio to classes
        self.effects.set_audio(self.path1, self.y1, self.sr1)
        self.features.set_audio(self.path1, self.y1, self.sr1)
        self.transitions.set_audio(self.path1, self.y1, self.sr1, self.path2, self.y2, self.sr2)