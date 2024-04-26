import asyncio
import queue
import subprocess
import threading
import time
from datetime import datetime

import speech_recognition as sr
from openai import OpenAI

from audio.processing.voice_to_text import voice_to_text
from audio.processing.voice_to_text_faster import voice_to_text_faster
from audio.processing.voice_to_text_mlx import voice_to_text_mlx

vtt_type = "MLX"  # MLX, FASTER_WHISPER, WHISPER
language_map = {
    "german": "de",
    "english": "en"
}
user_language = "german"
open_ai_server = "http://localhost:1234/v1"
llm_api_key = "not needed for a local LLM server"
llm_model = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf"
whisper_model_type = "small"

# blacklist of words that are wrongly recognized from speech to text but never spoken.
blacklist = ["Copyright", "WDR", "Thank you."]
now = datetime.now()
formatted_date = now.strftime("%d %B %Y")

talking_queue = queue.Queue()


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


personalitySystemPrompt = open_file("./ai/personalities/elsa.txt")
chat_messages = [{"role": "system",
                  "content":
                      personalitySystemPrompt + f" Deine Hauptsprache ist {user_language} und du antwotest immer auf "
                                                f"{user_language}."
                                                f"Das heutige Datum ist {formatted_date}."
                  }]

# -------------------------------------------------------------


openAiClient = OpenAI(api_key=llm_api_key, base_url=open_ai_server)

recognizer = sr.Recognizer()


def ask_open_ai_stream(messages):
    answer = ""
    answer_for_audio = ""
    for answer_part in openAiClient.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0,
            top_p=0.95,
            stream=True
    ):
        content = answer_part.choices[0].delta.content
        if content is not None:
            answer += content
            answer_for_audio += content
            if (answer_for_audio.endswith(".") and not answer_for_audio[-2].isdigit()
                    or answer_for_audio.endswith("?"
                                                 or
                                                 answer_for_audio.endswith("!"))):
                print(f"AI: {answer_for_audio}")
                talking_queue.put(lambda text=answer_for_audio: text_to_speech(text))
                answer_for_audio = ""
        else:
            print(f"AI: {answer_for_audio}")
            talking_queue.put(lambda text=answer_for_audio: text_to_speech(text))
    return answer


def not_black_listed(spoken1):
    for item in blacklist:
        if spoken1.__contains__(item):
            print(f"Blacklisted item found: {item}")
            return False
    return True


def text_to_speech(text):
    if text is not None and text.strip() != "":
        timeout = max(4, len(text.split(" ")))
        try:
            subprocess.run(['say', text], shell=False, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Timeout error while speaking. {timeout} seconds passed.")
            pass


def talking_worker():
    global STOP_TALKING
    while True:
        time.sleep(0.01)
        if not talking_queue.empty():
            task = talking_queue.get()
            task()
            talking_queue.task_done()


async def main():
    thread = threading.Thread(target=talking_worker)
    thread.daemon = True
    thread.start()
    global STOP_TALKING
    with sr.Microphone() as source:
        while True:
            try:
                if vtt_type == "MLX":
                    spoken = voice_to_text_mlx(source, language_map[user_language], whisper_model_type)
                elif vtt_type == "FASTER_WHISPER":
                    spoken = voice_to_text_faster(source, language_map[user_language], whisper_model_type)
                else:
                    spoken = voice_to_text(source, language_map[user_language], whisper_model_type)
                spoken = spoken.strip()
                spoken_lower = spoken.lower()
                print(spoken_lower != "" and len(spoken_lower.split(" ")) > 3 and not_black_listed(spoken))
                if spoken != "" and len(spoken.split(" ")) > 3 and not_black_listed(spoken):
                    if not talking_queue.empty():
                        talking_queue.queue.clear()
                        print(f"Stopped talking")
                    start_time = time.time()
                    ask_llm = spoken
                    chat_messages.append({"role": "user", "content": ask_llm})
                    answer = ask_open_ai_stream(chat_messages)
                    chat_messages.append({"role": "assistant", "content": answer.strip()})
                    stop_time = time.time()
                    print("LLM duration: " + str(stop_time - start_time))

            except sr.UnknownValueError:
                print("Sorry, I could not understand what you said.")
            except sr.RequestError as e:
                print("Sorry, an error occurred while trying to access the Google Web Speech API: {0}".format(e))


asyncio.run(main())
