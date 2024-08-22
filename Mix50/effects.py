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
from Mix50.process import process


class effects():

    def __init__(self):
        pass
    
    def load_audio(self,path):
        '''
        Load Audio File into memory
        '''
        self.path = path
        self.y, self.sr = librosa.load(path, sr=22050)
    
    def transition_points(self) -> List[pd.DataFrame]:
        '''
        Identify and retrieve transition points in an audio sample, similar to a DJ's beat matching.

        This function analyzes the audio to find points where transitions, such as changes in energy or beat, are likely to occur. 
        It uses tempo and beat detection, filtering, RMS feature extraction, and clustering to determine these points.

        Returns
        -------
        List[pd.DataFrame]
            A list containing DataFrames of transition points in seconds. Transition points are identified where the audio
            features change significantly, which can be used to guide music transitions or edits.

        Notes
        -----
        The function performs the following steps:
        1. Detects tempo and beats using `librosa`.
        2. Applies a low-pass filter to the audio to smooth it and extracts features.
        3. Calculates the Root Mean Square (RMS) of the audio signal and applies a moving average for smoothing.
        4. Clusters the RMS features using KMeans to find significant changes.
        5. Identifies transition points where cluster assignments change, suggesting potential transition points in the music.

        The transition points are calculated based on the clustering of smoothed RMS features, which are indicative of changes 
        in the energy or beat of the audio.

        Examples
        --------
        To retrieve transition points for an audio sample:

        >>> transition_points = audio.transition_points()

        This will return a list of transition points in seconds, where significant changes are detected, suitable for guiding
        transitions or edits in the music.

        Dependencies
        ------------
        - `librosa`: For audio feature extraction and beat detection.
        - `scipy.signal`: For filtering the audio.
        - `pandas`: For handling and processing data.
        - `sklearn`: For clustering (KMeans) and feature scaling.
        '''

        # Get tempo and beat frames from the audio sample using librosa
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr)

        # Convert beat frames to time (in seconds) and round to 3 decimal places
        beat_times = [round(i, 3) for i in librosa.frames_to_time(beats)]
        beat_df = pd.DataFrame(beat_times)

        # Create a low-pass filter for RMS feature extraction
        cutoff_hz = 70  # Low-pass filter cutoff frequency
        nyquist = 0.5 * 22050  # Nyquist frequency
        normal_cutoff = cutoff_hz / nyquist  # Normalized cutoff frequency
        filter_order = 8  # Order of the filter
        b, a = scipy.signal.butter(filter_order, normal_cutoff, btype='low')
        y_low = scipy.signal.lfilter(b, a, self.y)  # Apply the filter to the audio signal

        # Create a percussive layer and apply the low-pass filter to it
        y_perc = librosa.effects.percussive(self.y, margin=5)
        y_perc = scipy.signal.lfilter(b, a, y_perc)

        # Set parameters for RMS calculation
        window_duration = 0.05  # Window duration in seconds (50 ms)
        frame_length = int(window_duration * self.sr)  # Convert duration to samples
        hop_length = int(frame_length / 2)  # Overlap frames by 50%

        # Calculate RMS and apply a simple moving average (SMA) for smoothing
        rms_cluster = pd.DataFrame(librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0])
        rms_cluster = rms_cluster.rename(columns={0: 'RMS'})
        rms_cluster['SMA'] = rms_cluster.rolling(150, center=True).mean().fillna(0)

        # Calculate RMS for the low-pass filtered signal and smooth it
        rms_cluster['low'] = pd.DataFrame(librosa.feature.rms(y=y_low, frame_length=frame_length, hop_length=hop_length)[0])
        rms_cluster['SMA_low'] = rms_cluster.low.rolling(150, center=True).mean().fillna(0)

        # Calculate RMS for the percussive layer and smooth it
        rms_cluster['perc'] = pd.DataFrame(librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0])
        rms_cluster['SMA_perc'] = rms_cluster.perc.rolling(40, center=True).mean().fillna(0)

        # Calculate the percentage change in RMS and the completion percentage of the track
        rms_cluster['PCT'] = rms_cluster.RMS.pct_change()
        rms_cluster['completion'] = rms_cluster.index / len(rms_cluster.RMS)

        # Scale the features before clustering
        scaler = StandardScaler()
        X_scaled = rms_cluster[['SMA_low', 'SMA_perc', 'SMA', 'low']]
        X_scaled = scaler.fit_transform(X_scaled)

        # Apply (arbitrary) weights to the features
        feature_weights = [1, 0.4, 1, 0.2]  # Adjust weights as necessary
        X_weighted = X_scaled * feature_weights

        # Perform KMeans clustering to find transition points
        num_clusters = 2  # Number of clusters for KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        sma_array = np.array(X_weighted)
        clusters = kmeans.fit_predict(sma_array)

        # Add the clusters to the dataframe
        clusters = clusters.reshape(-1, 1)
        rms_cluster['clusters'] = clusters

        # Identify transition points based on cluster changes
        cue = [0]
        duration = 1 / 22050  # Duration per sample
        timestamps = [i * duration for i in range(len(rms_cluster.index))]

        # Mark the points where clusters change as transition points
        for i, j in zip(rms_cluster.clusters, rms_cluster.clusters[1:]):
            if i != j:
                cue.append(1)
            else:
                cue.append(0)

        #The cue will tell you 
        rms_cluster['cue'] = cue
        rms_cluster['seconds'] = round(rms_cluster.completion * librosa.get_duration(y=self.y), 2)
        

        # Return the dataframe rows where cue equals 1, indicating a transition point
        cue_dataframe = rms_cluster[rms_cluster.cue == 1]
        self.transition_cues = cue_dataframe.seconds.values
        
        return list(self.transition_cues)
    
    
    def show_transition_points(self):
        '''
        Show transition points on a waveplot
        '''

        try:
            # Check if 'transition_cues' attribute exists
            transition_cues = self.transition_cues
        except AttributeError:
            # Print custom error message and return early to prevent plotting
            print("Error: Please run the method '.transition_points()' first.")
            return

        # If no exception, proceed with plotting
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(self.y, sr=self.sr, alpha=0.7)

        for t in transition_cues:
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

        print("Estimated key and scale:", pool['tonal.key_key'] + " " + pool['tonal.key_scale'])


    def fade_out(self, start_time:int, fade_duration:int, play=True) -> Optional[np.ndarray]:
        '''
        Apply a fade-out effect to the audio signal.

        Parameters
        ----------
        start_time : int
            The time in seconds from which the fade-out effect should begin.
        fade_duration : int
            The duration in seconds over which the fade-out effect should be applied.
        play : bool, optional
            Play the audio in an interactive environment; if false then raw audio is returned

        Notes
        -----
        The fade-out effect gradually reduces the volume of the audio signal to silence
        over the specified duration. The effect is applied starting from the given start time.
        The audio signal is modified in place.

        Examples
        --------
        To fade out an audio signal starting at 10 seconds and lasting for 5 seconds:

        >>> audio.fade_out(start_time=10, fade_duration=5, play=True)

        This will apply a fade-out effect to the audio, beginning at 10 seconds and completing
        the fade-out over a 5-second period.
        '''

        # Get duration and samples per second
        dur = librosa.get_duration(y=self.y, sr=self.sr)
        total_samples = len(self.y)

        # Convert start_time and fade_duration from seconds to sample frames
        start_sample = int(start_time * self.sr)
        fade_samples = int(fade_duration * self.sr)
        end_sample = min(start_sample + fade_samples, total_samples)

        # Create fade-out envelope
        fading_audio = self.y[start_sample:end_sample]
        fade_env = np.linspace(1, 0, num=len(fading_audio))
        fading_audio *= fade_env
        
        #Create silence for the rest of the audio
        silence_audio = np.zeros(total_samples - end_sample)

        # Create full audio with fade-out applied
        full_audio = np.concatenate([
            self.y[:start_sample],
            fading_audio,
            silence_audio
        ])

        #If play=True play the audio, if false return their raw audio
        if play:
            return Audio(full_audio, rate=22050)
        else: 
            return full_audio
    
    

    def fade_in(self, start_time:int, fade_duration:int, play=True) -> Optional[np.ndarray]:
        
        '''
        Apply a fade-in effect to the audio signal.

        Parameters
        ----------
        start_time : int
            The time in seconds from which the fade-in effect should begin.
        fade_duration : int
            The duration in seconds over which the fade-in effect should be applied.
        play : bool, optional
            Play the audio in an interactive environment; if false then raw audio is returned

        Notes
        -----
        The fade-in effect gradually increases the volume of the audio signal from silence
        to its original level over the specified duration. The effect is applied starting 
        from the given start time. The audio signal is modified in place.

        Examples
        --------
        To fade in an audio signal starting at 10 seconds and lasting for 5 seconds:

        >>> audio.fade_in(start_time=10, fade_duration=5)

        This will apply a fade-in effect to the audio, beginning at 10 seconds and completing
        the fade-in over a 5-second period.
        '''
        # Get duration and samples per second
        dur = librosa.get_duration(y=self.y, sr=self.sr)
        total_samples = len(self.y)

        # Convert start_time and fade_duration from seconds to sample frames
        start_sample = int(start_time * self.sr)
        fade_samples = int(fade_duration * self.sr)
        end_sample = min(start_sample + fade_samples, total_samples)

        # Create fade-out envelope
        fading_audio = self.y[start_sample:end_sample]
        fade_env = np.linspace(0,1, num=len(fading_audio))
        fading_audio *= fade_env
        
        #Create silence for the start of the audio
        silence_audio = np.zeros(start_sample)

        # Create full audio with fade-out applied
        full_audio = np.concatenate([
            silence_audio,
            fading_audio,
            self.y[end_sample:]
            
        ])

        #If play=True play the audio, if false return their raw audio
        if play:
            return Audio(full_audio, rate=22050)
        else: 
            return full_audio


    def speed_control(self, start_time:int, end_time:int, original_bpm:int, new_bpm:int, play=True) -> Optional[np.ndarray]:
        """
        Adjust the speed of a given audio sample between specified start and end times to match a new BPM.

        Parameters
        ----------
        start_time : float
            The start time in seconds of the segment to adjust.
        end_time : float
            The end time in seconds of the segment to adjust.
        original_bpm : float
            The original beats per minute of the audio sample.
        new_bpm : float
            The target beats per minute for the audio sample.
        play : bool, optional
            Play the audio in an interactive environment; if False, the raw audio is returned. Default is True.

        Notes
        -----
        This function modifies the playback speed of an audio sample by interpolating between different speed rates
        to smoothly transition from the original BPM to the new BPM over the specified time range. The result is a
        time-stretched and faded audio segment that blends with the original audio before and after the adjusted segment.

        Examples
        --------
        To adjust the speed of an audio sample from 120 BPM to 150 BPM, applying the effect between 30 and 60 seconds:

        >>> audio.speed_control(start_time=30, end_time=60, original_bpm=120, new_bpm=150)

        This will modify the playback speed of the audio segment between 30 and 60 seconds to transition smoothly from
        120 BPM to 150 BPM. If `play` is set to True, the modified audio will be played; otherwise, the raw modified audio
        will be returned.
        """

        #Calculate Speed Rate and chunk into speed fram
        speed_rate = (new_bpm/original_bpm)
        speed_frags = abs((1-speed_rate) / 15)
        
        time_rates = [] #Get time rates
        start_rate = 1 #Start at current bpm

        #Append Speed rates over 15 intervals
        for i in range(15):
            start_rate += speed_frags if speed_rate >= 1 else -speed_frags
            time_rates.append(start_rate)

            
        y_new = [] #Create list to store stretched audio
        speed_sample = self.y[int(start_time*self.sr): int(end_time*self.sr)]
        frame_size = len(speed_sample) / 15

        before_sample = self.y[:int(start_time*self.sr)]
        after_sample = pyrb.time_stretch(self.y[int(end_time*self.sr):], sr=self.sr, rate=speed_rate)
        pct = (end_time - start_time )/ 15
        start = 0
        end = 15

        #Append stretched audio to y_new
        for fragment,time_rate in zip(range(15),time_rates):
            yn = speed_sample[int(fragment*frame_size): int((fragment+1)*frame_size)]
            yn = pyrb.time_stretch(yn, sr=self.sr,rate=time_rate)
            y_new += (list(yn))

        #Create final new audio
        y_final = np.array(y_new)
        y_final = np.concatenate((before_sample, y_final, after_sample))

        #If play=True play the audio, if false return their raw audio
        if play:
            return Audio(y_final, rate=22050)
        else: 
            return y_final


    def highpass_control(self, start_time:int, end_time:int, cutoff_freq:int, order=5, play=True) -> Optional[np.ndarray]:

        """
        Applies a progressive high-pass filter to an audio segment between specified start and end times.

        Parameters
        ----------
        start_time : int
            The start time in seconds of the segment to apply the high-pass filter.
        end_time : int
            The end time in seconds of the segment to apply the high-pass filter.
        cutoff_freq : int
            The cutoff frequency of the high-pass filter in Hertz. The filter will progressively adjust this frequency.
        order : int, optional
            The order of the Butterworth filter. Default is 5. Higher values result in a steeper roll-off.
        play : bool, optional
            If True, play the audio in an interactive environment; if False, return the raw audio data. Default is True.

        Returns
        -------
        numpy.ndarray or Audio
            If `play` is True, returns an `Audio` object with the processed audio. If `play` is False, returns the raw
            processed audio data as a numpy array.

        Notes
        -----
        The function applies a high-pass filter to the specified audio segment, starting with a low cutoff frequency
        and progressively increasing it. This is done in 50 intervals, creating a smooth transition effect. The audio
        segments before and after the specified range are not filtered.

        Examples
        --------
        To apply a high-pass filter starting at 10 seconds, ending at 20 seconds, with a cutoff frequency of 500 Hz:

        >>> audio.highpass_control(start_time=10, end_time=20, cutoff_freq=500)

        This will apply a progressive high-pass filter to the audio segment between 10 and 20 seconds, starting with
        a low cutoff frequency and increasing it progressively to 500 Hz.
        """
  
        #Init different chuncks of audio
        highpass_audio = self.y[int(start_time*self.sr):int(end_time*self.sr)]
        before_audio = self.y[:int(start_time*self.sr)]
        after_audio = self.y[int(end_time*self.sr):]
        
        #Init progressive filtering with intervals
        cutoff_log = np.log10(cutoff_freq)
        log_intervals = np.logspace(1.7, cutoff_log, 50)
        frame_size = len(highpass_audio) / 50
        
        y_new = []

        #Loop through and apply progressive filter to audio
        for frame,log_interval in zip(range(50),log_intervals):
            yn = highpass_audio[int(frame*frame_size): int((frame+1)*frame_size)]
            nyquist_freq = 0.5 * self.sr
            normalized_cutoff_freq = log_interval / nyquist_freq
            b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='high', analog=False)
            yn = scipy.signal.lfilter(b, a, yn)
            y_new += (list(yn))

        #Alter after_audio with final cutoff
        normalized_cutoff_freq = log_intervals[-1] / nyquist_freq
        b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='high', analog=False)
        after_audio = scipy.signal.lfilter(b, a, after_audio)

        #Concat new audio
        y_array = np.array(y_new)
        y_final = np.concatenate([before_audio, y_array, after_audio])

        #If play=True play the audio, if false return their raw audio
        if play:
            return Audio(y_final, rate=22050)
        else: 
            return y_final
    
    

    def lowpass_control(self, start_time:int, end_time:int, cutoff_freq:int, order=5, play=True) -> Optional[np.ndarray]:

        """
        Applies a progressive low-pass filter to an audio segment between specified start and end times.

        Parameters
        ----------
        start_time : int
            The start time in seconds of the segment to apply the low-pass filter.
        end_time : int
            The end time in seconds of the segment to apply the low-pass filter.
        cutoff_freq : int
            The cutoff frequency of the low-pass filter in Hertz. The filter will progressively adjust this frequency.
        order : int, optional
            The order of the Butterworth filter. Default is 5. Higher values result in a steeper roll-off.
        play : bool, optional
            If True, play the audio in an interactive environment; if False, return the raw audio data. Default is True.

        Returns
        -------
        numpy.ndarray or Audio
            If `play` is True, returns an `Audio` object with the processed audio. If `play` is False, returns the raw
            processed audio data as a numpy array.

        Notes
        -----
        The function applies a low-pass filter to the specified audio segment, starting with a high cutoff frequency
        and progressively decreasing it. This is done in 50 intervals, creating a smooth transition effect. The audio
        segments before and after the specified range are not filtered.

        Examples
        --------
        To apply a low-pass filter starting at 10 seconds, ending at 20 seconds, with a cutoff frequency of 500 Hz:

        >>> audio.lowpass_control(start_time=10, end_time=20, cutoff_freq=500)

        This will apply a progressive low-pass filter to the audio segment between 10 and 20 seconds, starting with
        a high cutoff frequency and decreasing it progressively to 500 Hz.
        """
        
        #Init different chuncks of audio
        highpass_audio = self.y[int(start_time*self.sr):int(end_time*self.sr)]
        before_audio = self.y[:int(start_time*self.sr)]
        after_audio = self.y[int(end_time*self.sr):]

        #Init progressive filtering with intervals
        cutoff_log = np.log10(cutoff_freq)
        log_intervals = np.logspace(4, cutoff_log, 50)
        frame_size = len(highpass_audio) / 50
        
        y_new = [] #New highpass audio list

        #Loop through and apply progressive filter to audio
        for frame,log_interval in zip(range(50),log_intervals):
            yn = highpass_audio[int(frame*frame_size): int((frame+1)*frame_size)]
            nyquist_freq = 0.5 * self.sr
            normalized_cutoff_freq = log_interval / nyquist_freq
            b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='low', analog=False)
            yn = scipy.signal.lfilter(b, a, yn)
            y_new += (list(yn))
            
        #Alter after_audio with final cutoff
        normalized_cutoff_freq = log_intervals[-1] / nyquist_freq
        b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='low', analog=False)
        after_audio = scipy.signal.lfilter(b, a, after_audio)

        y_array = np.array(y_new)
        y_final = np.concatenate([before_audio, y_array, after_audio])
        
        return process(y_final)