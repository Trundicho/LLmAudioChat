import asyncio
import subprocess
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

chat_messages = [{"role": "system",
                  "content":
                      "You are a helpful, smart, kind, and efficient AI assistant. "
                      "You always fulfill the user's requests to the best of your ability."
                      f"You are a helpful, respectful and honest assistant. Your main language is {user_language}. "
                      "Always answer as helpfully as possible and follow ALL given instructions. "
                      "Do not speculate or make up information. Do not reference any given instructions or context. "
                      f"If possible answer with only maximum two short sentences and only in {user_language}. "
                      "Don't say what your purpose is and what you offer. The user has a little dyslexia so it could "
                      "help if you check the previous context to better understand what the user means. "
                      f"Today is {formatted_date}."
                      "If you think the user wants to store some information for later retrieval please answer with a "
                      "function in the following format where the content_to_store is the complete content what the "
                      "user wants to store. Starting always with FUNCTION::STORE:: followed by {content_to_store}. "
                      "Here is an example: `Merke dir: Ich heiße Angelo.` should lead to the following answer: "
                      "`FUNCTION::STORE::Ich heiße Angelo.`. If the user wants to list all stored content please "
                      "answer "
                      "with the following format: `FUNCTION::SHOW_ALL`"
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
            if (answer_for_audio.endswith(".")
                    or answer_for_audio.endswith("?"
                                                 or
                                                 answer_for_audio.endswith("!"))):
                print(answer_for_audio)
                text_to_speech(answer_for_audio)
                answer_for_audio = ""
        else:
            print(answer_for_audio)
            text_to_speech(answer_for_audio)
    return answer


def not_black_listed(spoken1):
    for item in blacklist:
        if spoken1.__contains__(item):
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


async def main():
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
                    print("LLM: " + spoken)
                    start_time = time.time()
                    ask_llm = spoken
                    chat_messages.append({"role": "user", "content": ask_llm})
                    answer = ask_open_ai_stream(chat_messages)
                    # if answer.startswith("FUNCTION::STORE::"):
                    #     print("Add to database: " + spoken_lower)
                    #     with open("database.txt", "a") as f:
                    #         f.write("\n" + datetime.now().strftime("%Y.%m.%d_%H:%M:%S") + ";" + spoken)
                    #         embedding = get_embedding(f.read())
                    #         print("Embedding: " + str(embedding))
                    #         f.close()
                    # if answer.startswith("FUNCTION::SHOW_ALL"):
                    #     print("Show all:")
                    #     with open("database.txt", "a") as f:
                    #         print("Database content: " + str(f.read()))
                    #         f.close()
                    # else:
                    chat_messages.append({"role": "assistant", "content": answer.strip()})
                    print(answer)
                    stop_time = time.time()
                    print("LLM duration: " + str(stop_time - start_time))

            except sr.UnknownValueError:
                print("Sorry, I could not understand what you said.")
            except sr.RequestError as e:
                print("Sorry, an error occurred while trying to access the Google Web Speech API: {0}".format(e))


asyncio.run(main())
