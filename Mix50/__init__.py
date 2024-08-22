###############
#Import Dependencies
##############

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import pyrubberband as pyrb
from scipy import interpolate
from pydub import AudioSegment
from sklearn.cluster import KMeans
from IPython.display import Audio
import scipy
from essentia.standard import *
import essentia.streaming as ess
import essentia.standard as es
from sklearn.preprocessing import StandardScaler
from typing import List
import matplotlib.pyplot as plt
import sounddevice as sd
from typing import Optional, List