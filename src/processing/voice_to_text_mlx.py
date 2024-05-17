import time

import numpy as np
import speech_recognition as sr

from src.processing.voice_to_text_interface import IVoiceToText
from whispermlx import transcribe


class VoiceToTextMlx(IVoiceToText):
    def __init__(self):
        self.recognizer = sr.Recognizer()

    # Size	    Parameters	English-only    Multilingual    Required VRAM	Relative speed
    # tiny	    39 M	    tiny.en	        tiny	        ~1 GB	        ~32x
    # base	    74 M	    base.en	        base	        ~1 GB	        ~16x
    # small	    244 M	    small.en	    small	        ~2 GB	        ~6x
    # medium    769 M	    medium.en	    medium	        ~5 GB	        ~2x
    # large	    1550 M	    N/A	            large           ~10 GB	        1x
    def voice_to_text(self, source, language, whisper_model_type: str = "base"):
        print("Speak something...")
        audio = self.recognizer.listen(source)
        print("Recording complete.")
        start = time.time()
        audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        decode_options = {'language': language, 'fp16': False}
        text_ = transcribe(audio=audio_data, verbose=True, **decode_options,
                           path_or_hf_repo="mlx-community/whisper-" + whisper_model_type + "-mlx")['text']
        print("Transcribe duration: " + str((time.time() - start)))
        return text_

# with sr.Microphone() as source:
#     while True:
#         try:
#             spoken = voice_to_text_mlx(source, "de")
#             print(spoken)
#         except sr.UnknownValueError:
#             print("Sorry, I could not understand what you said.")
#         except sr.RequestError as e:
#             print("Sorry, an error occurred while trying to access the Google Web Speech API: {0}".format(e))
