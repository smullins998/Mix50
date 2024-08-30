#Work in progress
from Mix50.features import Features
from Mix50.effects import Effects
from Mix50.process import Process
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class Transitions:

    def __init__(self):
        
        pass


    def set_audio(self, path1, y1, sr1, path2, y2, sr2):
        '''
        Set the audio from our mixfifty module
        '''
        self.y1 = y1
        self.sr1 = sr1
        self.path1 = path1
        self.y2 = y2
        self.sr2 = sr2
        self.path2 = path2
        
        #We need both features sets and effects within our transitions
        #There is likely a better way to do this
        self.features_audio_1 = Features()
        self.features_audio_2 = Features()
        self.features_audio_1.set_audio(self.path1, self.y1,self.sr1)
        self.features_audio_2.set_audio(self.path2, self.y2,self.sr2)

        self.effects_audio_1 = Effects()
        self.effects_audio_2 = Effects()
        self.effects_audio_1.set_audio(self.path1, self.y1,self.sr1)
        self.effects_audio_2.set_audio(self.path2, self.y2,self.sr2)


    def crossfade(self, 
                  cue_num1:int, 
                  cue_num2:int,
                  filter_type:str, 
                  fade_duration:int=15, 
                  cutoff_frequency:int=0) -> object:
        """
    Apply speed control, filtering, and crossfade effects to two audio tracks based on their beat grids.

    This method processes two audio tracks by aligning their beats, adjusting their speeds to match BPMs, 
    applying specified filters, and performing a crossfade transition between them. The resulting audio 
    is a seamless mix of both tracks with the desired effects applied.

    Parameters:
        cue_num1 (int):
            The cue number for the first audio track. 
            Determines the start time for applying speed control and other effects.
        
        cue_num2 (int):
            The cue number for the second audio track. 
            Determines the start time for the fade in of the second track
        
        filter_type (str):
            The type of filter to apply to the second audio track. 
            Must be one of the following:
                - 'highpass': Apply a high-pass filter.
                - 'lowpass': Apply a low-pass filter.
                - 'none': Do not apply any filter.
        
        fade_duration (int, optional):
            The duration of the transition effect in seconds. 
            Defaults to 15 seconds.
        
        cutoff_frequency (int, optional):
            The cutoff frequency for the filter in Hertz (Hz). 
            Defaults to 0 Hz.

    Returns:
        Process:
            An instance of the `Process` class containing the final mixed audio 
            after applying speed control, filtering, and crossfade effects.

    Raises:
        IndexError:
            If `cue_num1` or `cue_num2` is out of range of the available loop cues 
            in the beat grids of the audio tracks.
        
        ValueError:
            If an invalid `filter_type` is provided. 
            Valid options are 'highpass', 'lowpass', or 'none'.

        """
        
        # Retrieve audio features and data
        beats1, beats2 = self.features_audio_1.beats(), self.features_audio_2.beats()
        bpm1, bpm2 = self.features_audio_1.bpm(), self.features_audio_2.bpm()
        y1, y2 = self.y1, self.y2
        sr1, sr2 = self.sr1, self.sr2

        # Define constants
        fade_duration2 = fade_duration
        sample_rate = 22050
        cutoff_frequency = cutoff_frequency
        
        # Determine start and end times for the crossfade. Use the cue number and raise if out of range
        try:
            loop_cue_values2 = beats2[beats2.loop_cues.notna()].loop_cues.values.tolist()
            loop_cue_values1 = beats1[beats1.loop_cues.notna()].loop_cues.values.tolist()    
            start_time2 = loop_cue_values2[cue_num2]
            end_time2 = loop_cue_values2[cue_num2 + 2]
            start_time1 = loop_cue_values1[cue_num1]
        except IndexError:
            raise IndexError("The cue number is out of range. Please pick a smaller cue number.")

        
        # Apply speed control, high-pass filter, and fade out effects
        audio2 = self.effects_audio_2.speed_control(start_time2, end_time2, bpm2, bpm1).raw_audio()
        self.effects_audio_2.set_audio('./', audio2, sample_rate)

        #Set filter types
        if filter_type == 'highpass':
            audio2 = self.effects_audio_2.highpass_control(start_time2, end_time2, cutoff_frequency).raw_audio()
            self.effects_audio_2.set_audio('./', audio2, sample_rate)
        elif filter_type == 'lowpass':
            audio2 = self.effects_audio_2.lowpass_control(start_time2, end_time2, cutoff_frequency).raw_audio()
            self.effects_audio_2.set_audio('./', audio2, sample_rate)
        elif filter_type == 'none':
            pass
        
        else:
            raise ValueError("Invalid filter type: please choose either 'highpass' or 'lowpass'")

        audio2 = self.effects_audio_2.fade_out(end_time2, fade_duration2).raw_audio() #Start fading at the end of the FX

        #Now that we have our affected audio for song2, we need to calculate a new beatgrid because of altered speed
        #Let's first return the beatgrid starting from our start_transition_time
        affected_beatgrid_2 = beats2[(beats2.beats >= start_time2) & (beats2.beats <= end_time2)]
        
        #Get relevent beats to slice our DF
        last_beat = affected_beatgrid_2.beats.values[-1]
        first_beat = affected_beatgrid_2.beats.values[0]
        time_between_first_last = last_beat - first_beat
        last_beat_affected = time_between_first_last * (bpm2/bpm1) + first_beat
        num_beats = len(affected_beatgrid_2)
        new_beats = np.linspace(first_beat, last_beat_affected, num=num_beats)
        
        # Replace old beats with new beats
        affected_beatgrid_2['beats'] = new_beats
        
        # Update downbeat, loop_cues, and transitions based on new beats
        # The conditions remain the same as they are just updates based on new timing
        affected_beatgrid_2['downbeat'] = np.where(affected_beatgrid_2['downbeat'].notna(), affected_beatgrid_2['beats'], affected_beatgrid_2['downbeat'])
        affected_beatgrid_2['loop_cues'] = np.where(affected_beatgrid_2['loop_cues'].notna(), affected_beatgrid_2['beats'], affected_beatgrid_2['loop_cues'])
        affected_beatgrid_2['transitions'] = np.where(affected_beatgrid_2['transitions'].notna(), affected_beatgrid_2['beats'], affected_beatgrid_2['transitions'])
        
        #Lets get the before-effects beatgrid
        before_affected_beatgrid_2 = beats2[(beats2.beats < start_time2)]
        after_affected_beatgrid_2 = beats2[(beats2.beats > end_time2)]
        
        #Now we have to affect the beatgrid after the transition
        first_beat = after_affected_beatgrid_2.beats.values[0]
        second_beat = after_affected_beatgrid_2.beats.values[1]
        time_between_first_second = second_beat - first_beat
        speed_factor = time_between_first_second * (bpm2/bpm1) #New time between beats accounting for speed change
        num_beats = len(after_affected_beatgrid_2)
        new_beats = [first_beat + (speed_factor*frame) for frame in range(num_beats)]
        
        # Replace old beats with new beats
        after_affected_beatgrid_2['beats'] = new_beats
        
        # Update downbeat, loop_cues, and transitions based on new beats
        # The conditions remain the same as they are just updates based on new timing
        after_affected_beatgrid_2['downbeat'] = np.where(after_affected_beatgrid_2['downbeat'].notna(), after_affected_beatgrid_2['beats'], after_affected_beatgrid_2['downbeat'])
        after_affected_beatgrid_2['loop_cues'] = np.where(after_affected_beatgrid_2['loop_cues'].notna(), after_affected_beatgrid_2['beats'], after_affected_beatgrid_2['loop_cues'])
        after_affected_beatgrid_2['transitions'] = np.where(after_affected_beatgrid_2['transitions'].notna(), after_affected_beatgrid_2['beats'], after_affected_beatgrid_2['transitions'])
        
        #Concat final beatgrid with effects taken place
        final_affected_beatgrid = pd.concat([before_affected_beatgrid_2, affected_beatgrid_2, after_affected_beatgrid_2])

        #Now we have the affected 2nd song, we need to overlay the first.
        new_start_time2 = final_affected_beatgrid[final_affected_beatgrid.loop_cues.notna()].loop_cues.values.tolist()[cue_num2]
        new_end_time2 = final_affected_beatgrid[final_affected_beatgrid.loop_cues.notna()].loop_cues.values.tolist()[cue_num2]
        
        #Break the audio into pieces
        audio2_overlay_len = int((new_end_time2+fade_duration2)*sample_rate)
        audio2_end_sample = audio2_overlay_len+((fade_duration2+5)*sample_rate)
        audio2_beginning_overlay_sample = int((new_end_time2+fade_duration2)*sample_rate)
        audio2_beginning = audio2[:int((new_end_time2+fade_duration2)*sample_rate)]
        audio2_overlay = audio2[audio2_beginning_overlay_sample:audio2_end_sample]
        
        #Get start time of transition song and overlay with other song
        audio1_sample_start = int(start_time1*sample_rate)
        audio1_overlay = y1[audio1_sample_start:audio1_sample_start+((fade_duration2+5)*sample_rate)]
        audio1_overlay = audio1_overlay * np.linspace(0,1,len(audio1_overlay)) #Fade the second song in. Without this it is abrupt
        audio1_end = y1[audio1_sample_start+((fade_duration2+5)*sample_rate):]
        
        #Full overlay 
        full_overlay = audio2_overlay + audio1_overlay
        full_audio = list(audio2_beginning) + list(full_overlay) + list(audio1_end)

        return Process(full_audio)