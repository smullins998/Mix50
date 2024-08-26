import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment
from sklearn.cluster import KMeans
from IPython.display import Audio
import scipy
from essentia.standard import *
import essentia.streaming as ess
import essentia.standard as es
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sounddevice as sd
from typing import Optional, List
from Mix50.process import Process

    
class Features:
    
    def __init__(self):

        pass
    
    
    def set_audio(self, path, y1, sr1):
        '''
        Set the audio from our mixfifty module
        '''
        self.y = y1
        self.sr = sr1
        self.path = path
  
    
    def beats(self) -> pd.DataFrame:
        """
        Identify transition points in an audio sample, similar to DJ's beat matching.

        Analyzes the audio to find points where transitions are likely to occur based on changes in energy or beat. 
        Uses tempo and beat detection, filtering, RMS feature extraction, and clustering to identify these points.

        Returns
        -------
        pd.DataFrame
            DataFrame of transition points in seconds. Includes columns for beats, downbeats, loop cues, and transitions.
        """

        # Detect tempo and beats using librosa
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)
        beat_times = [round(i, 3) for i in librosa.frames_to_time(beats)]

        # Low-pass filter setup
        cutoff_hz = 70
        nyquist = 0.5 * self.sr
        normal_cutoff = cutoff_hz / nyquist
        b, a = scipy.signal.butter(8, normal_cutoff, btype='low')

        # Apply low-pass filter to audio
        y_low = scipy.signal.lfilter(b, a, self.y)
        y_perc = librosa.effects.percussive(self.y, margin=5)
        y_perc = scipy.signal.lfilter(b, a, y_perc)

        # RMS and smoothing parameters
        window_duration = 0.05
        frame_length = int(window_duration * self.sr)
        hop_length = frame_length // 2

        # RMS calculation and smoothing
        rms_df = pd.DataFrame({
            'RMS': librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0],
            'SMA': pd.Series(librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0]).rolling(150, center=True).mean().fillna(0),
            'low': librosa.feature.rms(y=y_low, frame_length=frame_length, hop_length=hop_length)[0],
            'SMA_low': pd.Series(librosa.feature.rms(y=y_low, frame_length=frame_length, hop_length=hop_length)[0]).rolling(150, center=True).mean().fillna(0),
            'perc': librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0],
            'SMA_perc': pd.Series(librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0]).rolling(40, center=True).mean().fillna(0),
        })

        # Calculate percentage change and completion
        rms_df['PCT'] = rms_df['RMS'].pct_change()
        rms_df['completion'] = rms_df.index / len(rms_df)

        # Scale and weight features
        scaler = StandardScaler()
        features = ['SMA_low', 'SMA_perc', 'SMA', 'low']
        X_scaled = scaler.fit_transform(rms_df[features])
        X_weighted = X_scaled * [1, 0.4, 1, 0.2]

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        rms_df['clusters'] = kmeans.fit_predict(X_weighted)

        # Identify transition points
        rms_df['cue'] = rms_df['clusters'].diff().ne(0).astype(int)
        rms_df['seconds'] = round(rms_df['completion'] * librosa.get_duration(y=self.y), 2)

        # Retrieve cues and find closest values to beats
        onset = rms_df[rms_df.cue == 1]['seconds'].values
        if len(onset) == 0:
            return pd.DataFrame(columns=['beats', 'downbeat', 'loop_cues', 'transitions'])

        differences = [abs(value - onset[0]) for value in beat_times]
        idx_pos = differences.index(min(differences))
        closest_value = beats[idx_pos]

        # Calculate downbeats, loop cues, and transitions
        downbeat = beat_times[idx_pos::-4][::-1][:-1] + beat_times[idx_pos::4]
        loop_cues = beat_times[idx_pos::-16][::-1][:-1] + beat_times[idx_pos::16]
        transitions = [min(loop_cues, key=lambda x: abs(x - value)) for value in onset]

        # Create beat DataFrame with results
        beat_df = pd.DataFrame({'beats': beat_times})
        beat_df['downbeat'] = beat_df['beats'].apply(lambda x: x if x in downbeat else None)
        beat_df['loop_cues'] = beat_df['beats'].apply(lambda x: x if x in loop_cues else None)
        beat_df['transitions'] = beat_df['beats'].apply(lambda x: x if x in transitions else None)
        self.beat_df = beat_df
        
        return beat_df
    
    
    def transition_points(self) -> np.ndarray:
        '''
        Retrieve transition points from the processed beat DataFrame.

        This method extracts transition points from the `beat_df` DataFrame, which should be populated by the `.beats()` method.
        Transition points represent significant changes or cues in the audio data that were identified during the beat analysis.

        Returns
        -------
        np.ndarray
            An array of transition points, which are significant moments in the audio where transitions occur. 
            NaN values are excluded from the output.

        Raises
        ------
        AttributeError
            If the `beat_df` attribute is not set (i.e., if `.beats()` has not been called before this method), an error message is printed
            and the method returns an empty list.

        Notes
        -----
        - Ensure that the `.beats()` method is executed before calling this method to populate the `beat_df` attribute.
        - The `beat_df` DataFrame is expected to have a column named 'transitions' which contains the transition points.

        Examples
        --------
        To retrieve transition points after running the `.beats()` method:

        >>> audio.beats()  # First, compute beats and transitions
        >>> transitions = audio.transition_points()  # Retrieve transition points
        >>> print(transitions)
        [12.34, 56.78, ...]

        '''

        try:
            # Check if data exists
            beat_df = self.beat_df
        except AttributeError:
            # Print custom error message and return early to prevent plotting
            print("Error: Please run the method '.beats()' first.")
            return
        
        transition_points = beat_df.transitions.values
        transition_points = transition_points[~np.isnan(transition_points)]
        
        return list(transition_points)
            
    
    def show_transition_points(self):
        '''
        Show transition points on a waveplot
        '''

        try:
            # Check if data exists
            beat_df = self.beat_df
        except AttributeError:
            # Print custom error message and return early to prevent plotting
            print("Error: Please run the method '.beats()' first.")
            return

        # If no exception, proceed with plotting
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.7)

        for t in list(beat_df.transitions):
            plt.axvline(x=t, color='r', linestyle='--', label='Transition Point')

        plt.title('Waveplot with Transition Points')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend(['Transition Point'], loc='upper right')
        plt.show()
        
        
    def beat_matching(self) -> pd.DataFrame:
        '''
        Retrieve downbeats using MonoLoader 
        '''
        
        #Load Audio
        audio = MonoLoader(filename=self.path)()

        # Compute beat positions and BPM
        rhythm_extractor = RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)

        beat_df = pd.DataFrame({'beats':beats})

        return beat_df


    def bpm(self) -> int:
        '''
        Retrieve BPM using MonoLoader 
        '''

        #Load audio
        audio = es.MonoLoader(filename=self.path)()

        # Compute beat positions and BPM.
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        
        return round(bpm,0)
    

    def key(self) -> str:
        '''
        Retrieve Key using Essentia 
        '''

        # Initialize Monoloader
        loader = ess.MonoLoader(filename=self.path)
        framecutter = ess.FrameCutter(frameSize=4096, hopSize=2048, silentFrames='noise')
        windowing = ess.Windowing(type='blackmanharris62')
        spectrum = ess.Spectrum()
        spectralpeaks = ess.SpectralPeaks(orderBy='magnitude',
                                          magnitudeThreshold=0.00001,
                                          minFrequency=20,
                                          maxFrequency=3500,
                                          maxPeaks=60)

        hpcp = ess.HPCP()
        hpcp_key = ess.HPCP(size=36, # We will need higher resolution for Key estimation.
                            referenceFrequency=440, # Assume tuning frequency is 44100.
                            bandPreset=False,
                            minFrequency=20,
                            maxFrequency=1500,
                            weightType='cosine',
                            nonLinear=False,
                            windowSize=1.)

        key = ess.Key(profileType='edma', # Use profile for electronic music.
                      numHarmonics=4,
                      pcpSize=36,
                      slope=0.6,
                      usePolyphony=True,
                      useThreeChords=True)

        # Use pool to store data.
        pool = essentia.Pool()

        # Connect streaming algorithms.
        loader.audio >> framecutter.signal
        framecutter.frame >> windowing.frame >> spectrum.frame
        spectrum.spectrum >> spectralpeaks.spectrum
        spectralpeaks.magnitudes >> hpcp.magnitudes
        spectralpeaks.frequencies >> hpcp.frequencies
        spectralpeaks.magnitudes >> hpcp_key.magnitudes
        spectralpeaks.frequencies >> hpcp_key.frequencies
        hpcp_key.hpcp >> key.pcp
        hpcp.hpcp >> (pool, 'tonal.hpcp')
        key.key >> (pool, 'tonal.key_key')
        key.scale >> (pool, 'tonal.key_scale')
        key.strength >> (pool, 'tonal.key_strength')

        essentia.run(loader)

        return "Estimated key and scale:", pool['tonal.key_key'] + " " + pool['tonal.key_scale']