from .effects import Effects
from .features import Features
from .process import Process


class Transitions():
    
    def __init__(self):
        
        pass
    
    
    def set_audio(self, path1, y1, sr1, path2, y2=None, sr2=None):
        '''
        Set the audio from our mixfifty module
        '''
        #Set variables
        self.y1 = y1
        self.sr1 = sr1
        self.path1 = path1
        self.y2 = y2
        self.sr2 = sr2
        self.path2 = path2
  
        #Create feature instances to use for our transition calculations
        #Set audio for the feature conditions. Is there a better way to do this?
        self.features1 = Features()
        self.features2 = Features()
        self.features1.set_audio(path1,y1,sr1)
        self.features2.set_audio(path2,y2,sr2)
        
        self.effects1 = Effects()
        self.effects2 = Effects()
        self.effects1.set_audio(path1,y1,sr1)
        self.effects2.set_audio(path2,y2,sr2)
        
    
    def crossfade_highpass(self):
        
        
        #EDITING THE FIRST SONG --> speeds it up and fades it out
        
        start_time2 = marker2[marker2.loop_cues.notna()].loop_cues.values.tolist()[-4]
        end_time2 = marker2[marker2.loop_cues.notna()].loop_cues.values.tolist()[-2]
        fade_dur2 = 20
        bpm2 = marker2.bpm[0]
        bpm1 = marker1.bpm[0]

        audio2 = speed_control(y2, sr2, start_time2, end_time2, bpm2, bpm1)
        audio2 = highpass_control(audio2,sr2, start_time2, end_time2, 250,5)
        audio2 = fade_out(audio2,end_time2, fade_dur2)

        #EDITING THE SECOND SONG
        fade_dur1 = 10

        audio1 = fade_in(y1,0.00,fade_dur1)

        df3 = markers(audio2)
        segment2 = df3[df3.loop_cues >= start_time2].loop_cues.values.tolist()[1]

        part_2_beg = list(audio2[:int(22050 * segment2)])
        part_2_end = list(audio2[int(22050 * segment2):])

        part_2_end_len = len(part_2_end)


        part_1_beg = list(audio1[:int(part_2_end_len)])
        part_1_end = list(audio1[int(part_2_end_len):])

        overlay = np.array(part_2_end) + np.array(part_1_beg)
        full_audio = list(part_2_beg) + list(overlay) + list(part_1_end)

        return full_audio