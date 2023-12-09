import speech_recognition as sr
import numpy as np
from whispermlx import transcribe
import time

recognizer = sr.Recognizer()

# Size	    Parameters	English-only    Multilingual    Required VRAM	Relative speed
# tiny	    39 M	    tiny.en	        tiny	        ~1 GB	        ~32x
# base	    74 M	    base.en	        base	        ~1 GB	        ~16x
# small	    244 M	    small.en	    small	        ~2 GB	        ~6x
# medium    769 M	    medium.en	    medium	        ~5 GB	        ~2x
# large	    1550 M	    N/A	            large           ~10 GB	        1x


def voice_to_text_mlx(source, language, whisper_model_type: str = "base"):
    print("Speak something...")
    audio = recognizer.listen(source)
    print("Recording complete.")
    start = time.time()
    audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    decode_options = {'language': language, 'fp16': False}
    text_ = transcribe(audio=audio_data, model=whisper_model_type, verbose=True, **decode_options)['text']
    print("Transcribe duration: " + str((time.time()-start)))
    return text_

# with sr.Microphone() as source:
#     while True:
#         try:
#             spoken = voice_to_text(source, "de")
#             print(spoken)
#         except sr.UnknownValueError:
#             print("Sorry, I could not understand what you said.")
#         except sr.RequestError as e:
#             print("Sorry, an error occurred while trying to access the Google Web Speech API: {0}".format(e))
