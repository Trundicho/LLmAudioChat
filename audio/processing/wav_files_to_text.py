import os
import wave
import time
from audio.processing.voice_to_text_mlx import voice_to_text

folder_path = '/path/to/wav/files'


def timer(fn, *args):
    for _ in range(5):
        fn(*args)

    num_its = 10

    tic = time.perf_counter()
    for _ in range(num_its):
        fn(*args)
    toc = time.perf_counter()
    return (toc - tic) / num_its


for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)
        with wave.open(file_path, 'rb') as wav_file:
            print(f"File: {filename}")
            print(f"Duration (seconds): {wav_file.getnframes() / float(wav_file.getframerate())}")

            start = time.time()
            text = voice_to_text(file_path, language="de", whisper_model_type="medium")
            print(filename + " :: " + text['text'])
            print("Duration: " + str((time.time()-start)))

