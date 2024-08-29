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
import sounddevice as sd
from typing import Optional, List
from Mix50.process import Process
from Mix50.features import Features

class Effects:

    def __init__(self):
        pass


    def set_audio(self, path, y1, sr1, y2=None, sr2=None):
        '''
        Set the audio from our mixfifty module
        '''
        self.y = y1
        self.sr = sr1
        self.path = path
        

    def __fade_between(self, sample, fade_in_duration, fade_out_duration):
        
        '''
        A helper function to fade between small samples of audio;
        Clicks and pops often occur when overlaying pieces of audio.
        This will prevent that by applying linesplace fade between overlays. 
        '''
        sample_length = len(sample)
        time = np.linspace(0, 1, sample_length)

        fade_in_env = np.linspace(0, 1, int(sample_length * fade_in_duration))
        fade_out_env = np.linspace(1, 0, int(sample_length * fade_out_duration))

        sample[:len(fade_in_env)] *= fade_in_env
        sample[-len(fade_out_env):] *= fade_out_env

        return sample


    def fade_out(self, start_time:int, fade_duration:int) -> object:
        '''
        Apply a fade-out effect to the audio signal.

        Parameters
        ----------
        start_time : int
            The time in seconds from which the fade-out effect should begin.
        fade_duration : int
            The duration in seconds over which the fade-out effect should be applied.

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

        #Error handling for OOB time arguments
        if len(fading_audio) == 0:
            raise ValueError("The start time is out of bounds; please choose a start time within the length of the audio")

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

        return Process(full_audio)
    

    def fade_in(self, start_time:int, fade_duration:int) -> object:
        
        '''
        Apply a fade-in effect to the audio signal.

        Parameters
        ----------
        start_time : int
            The time in seconds from which the fade-in effect should begin.
        fade_duration : int
            The duration in seconds over which the fade-in effect should be applied.

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
        
        #Error handling for OOB time arguments
        if len(fading_audio) == 0:
            raise ValueError("The start time is out of bounds; please choose a start time within the length of the audio")

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

        return Process(full_audio)


    def speed_control(self, start_time:int, end_time:int, original_bpm:int, new_bpm:int) -> object:
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
        num_fragments = 15 #Too high fragments will create clicks/pops

        #Calculate Speed Rate and chunk into speed fram
        speed_rate = (new_bpm/original_bpm)
        speed_frags = abs((1-speed_rate) / num_fragments)
        
        time_rates = [] #Get time rates
        start_rate = 1 #Start at current bpm

        #Append Speed rates over 15 intervals
        for i in range(num_fragments):
            start_rate += speed_frags if speed_rate >= 1 else -speed_frags
            time_rates.append(start_rate)

            
        y_new = [] #Create list to store stretched audio
        speed_sample = self.y[int(start_time*self.sr): int(end_time*self.sr)]
        frame_size = len(speed_sample) / num_fragments

        before_sample = self.y[:int(start_time*self.sr)]
        after_sample = pyrb.time_stretch(self.y[int(end_time*self.sr):], sr=self.sr, rate=speed_rate)
        pct = (end_time - start_time )/ num_fragments
        start = 0
        end = num_fragments

        #Append stretched audio to y_new
        for fragment,time_rate in zip(range(num_fragments),time_rates):
            yn = speed_sample[int(fragment*frame_size): int((fragment+1)*frame_size)]
            yn = pyrb.time_stretch(yn, sr=self.sr,rate=time_rate)
            yn = self.__fade_between(yn, .003, .003) #Fade between as to prevent pops/clicks
            y_new += (list(yn))

        #Create final new audio
        y_final = np.array(y_new)
        y_final = np.concatenate((before_sample, y_final, after_sample))

        return Process(y_final)


    def highpass_control(self, start_time:int, end_time:int, cutoff_freq:int, order=5) -> object:

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
        num_fragments = 50
  
        #Init different chuncks of audio
        highpass_audio = self.y[int(start_time*self.sr):int(end_time*self.sr)]
        before_audio = self.y[:int(start_time*self.sr)]
        after_audio = self.y[int(end_time*self.sr):]
        
        #Init progressive filtering with intervals
        cutoff_log = np.log10(cutoff_freq)
        log_intervals = np.logspace(1.7, cutoff_log, num_fragments)
        frame_size = len(highpass_audio) / num_fragments
        
        y_new = []

        #Loop through and apply progressive filter to audio
        for frame,log_interval in zip(range(num_fragments),log_intervals):
            yn = highpass_audio[int(frame*frame_size): int((frame+1)*frame_size)]
            nyquist_freq = 0.5 * self.sr
            normalized_cutoff_freq = log_interval / nyquist_freq
            b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='high', analog=False)
            yn = scipy.signal.lfilter(b, a, yn)
            yn = self.__fade_between(yn, .003, .003) #Fade between as to prevent pops/clicks
            y_new += (list(yn))

        #Alter after_audio with final cutoff
        normalized_cutoff_freq = log_intervals[-1] / nyquist_freq
        b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='high', analog=False)
        after_audio = scipy.signal.lfilter(b, a, after_audio)

        #Concat new audio
        y_array = np.array(y_new)
        y_final = np.concatenate([before_audio, y_array, after_audio])

        return Process(y_final)
    
    

    def lowpass_control(self, start_time:int, end_time:int, cutoff_freq:int, order=5) -> object:

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
        num_fragments = 50
        
        #Init different chuncks of audio
        highpass_audio = self.y[int(start_time*self.sr):int(end_time*self.sr)]
        before_audio = self.y[:int(start_time*self.sr)]
        after_audio = self.y[int(end_time*self.sr):]

        #Init progressive filtering with intervals
        cutoff_log = np.log10(cutoff_freq)
        log_intervals = np.logspace(4, cutoff_log, num_fragments)
        frame_size = len(highpass_audio) / num_fragments
        
        y_new = [] #New highpass audio list

        #Loop through and apply progressive filter to audio
        for frame,log_interval in zip(range(num_fragments),log_intervals):
            yn = highpass_audio[int(frame*frame_size): int((frame+1)*frame_size)]
            nyquist_freq = 0.5 * self.sr
            normalized_cutoff_freq = log_interval / nyquist_freq
            b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='low', analog=False)
            yn = scipy.signal.lfilter(b, a, yn)
            yn = self.__fade_between(yn, .003, .003) #Fade between as to prevent pops/clicks
            y_new += (list(yn))
            
        #Alter after_audio with final cutoff
        normalized_cutoff_freq = log_intervals[-1] / nyquist_freq
        b, a = scipy.signal.butter(order, normalized_cutoff_freq, btype='low', analog=False)
        after_audio = scipy.signal.lfilter(b, a, after_audio)

        y_array = np.array(y_new)
        y_final = np.concatenate([before_audio, y_array, after_audio])
        
        return Process(y_final)