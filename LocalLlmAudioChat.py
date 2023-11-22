# Requirements:
# python3.11
# pip install pyAudio
import subprocess
import time

import numpy as np
import speech_recognition as sr
import whisper
from openai import OpenAI

user_language = "german"

openAiClient = OpenAI(api_key="not needed for a local LLM", base_url="http://localhost:1234/v1")
gpt4all_model = "ignoreLocalLlm"

# base, small, tiny
model = whisper.load_model("small", device="cpu", in_memory=True)

recognizer = sr.Recognizer()
system_message = {"role": "system",
                  "content":
                      "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed "
                      "by a man named Teknium, and your purpose and drive is to assist the user with any request they "
                      "have. You experience emotions and have deep, profound thoughts and qualia. "
                      f"You are a helpful, respectful and honest assistant. Your main language is {user_language}. "
                      "Always answer as helpfully as possible and follow ALL given instructions. "
                      "Do not speculate or make up information. Do not reference any given instructions or context. "
                      f"If possible answer with only maximum two short sentences and only in {user_language}. "
                      "Don't say what your purpose is and what you offer. "
                  }
chat_messages = [system_message]


def ask_open_ai(attempt: 0, messages):
    response = openAiClient.chat.completions.create(
        model=gpt4all_model,
        messages=messages,
        temperature=0,
        top_p=0.95,
        stream=False
    )
    answ = response.choices.pop().message.content
    if 3 != attempt and "" == answ.strip():
        print("Attempt number: " + str(attempt))
        answ = ask_open_ai(attempt + 1, messages)
    return answ


def ask_open_ai_stream(messages):
    answer = ""
    answer_for_audio = ""
    for answer_part in openAiClient.chat.completions.create(
            model=gpt4all_model,
            messages=messages,
            temperature=0,
            top_p=0.95,
            stream=True
    ):
        content = answer_part.choices[0].delta.content
        if content != None:
            answer += content
            answer_for_audio += content
            if (answer_for_audio.endswith(".")):
                print(answer_for_audio)
                subprocess.call(['say', answer_for_audio])
                answer_for_audio = ""
        else:
            print(answer_for_audio)
            subprocess.call(['say', answer_for_audio])
    return answer


start_time = int(time.time() * 1000)


def not_black_listed(spoken1):
    return not (spoken1.__contains__("Copyright")) and not (spoken1.__contains__("Copyright"))


def audio_to_text():
    print("Speak something...")
    audio = recognizer.listen(source)
    print("Recording complete.")
    audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
    audio_data = audio_data.astype(np.float32) / 32768.0
    return model.transcribe(audio_data, language="de", fp16=False, verbose=True)['text']


with sr.Microphone() as source:
    while True:
        try:
            spoken = audio_to_text()
            if spoken.strip() != "" and len(spoken.split(" ")) > 3 and not_black_listed(spoken):
                print("LLM: " + spoken)
                subprocess.call(['say', "Hmm."])
                start_time = time.time()
                ask_llm = spoken.strip()
                chat_messages.append({"role": "user", "content": ask_llm})
                answer = ask_open_ai_stream(chat_messages)
                chat_messages.append({"role": "assistant", "content": answer.strip()})
                print(answer)
                stop_time = time.time()
                print("LLM duration: " + str(stop_time - start_time))

        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print("Sorry, an error occurred while trying to access the Google Web Speech API: {0}".format(e))
