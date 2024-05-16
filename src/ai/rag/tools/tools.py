import json
import re
import time
import webbrowser
from datetime import datetime

import pyautogui
import torch
from sentence_transformers import util

from src.ai.clients.ai_client_factory import AiClientFactory
from src.ai.clients.gemini_client import GeminiClient
from src.ai.clients.llm_vision import LlmVision
from src.ai.rag.tools.audiochat_functions import AudioChatFunctions
from src.audio_chat_config import AudioChatConfig
from src.tools.search.search import DuckDuckGoSearch
from src.tools.timer.timer import Timer


class Tools:
    def __init__(self, rag_vault_file="vault.txt", tts=None):
        tools_system_message_file = AudioChatConfig().get_config()["AI_CONFIG"]["FUNCTIONS_SYSTEM_MESSAGE"]
        self.rag_vault_file = rag_vault_file
        self.vault_updated = False
        self.embeddings_client = AiClientFactory().create_embeddings_client()
        self.timer = Timer()
        self.vision = LlmVision(tts)
        self.audio_chat_functions = AudioChatFunctions()
        self.functions = [
            self.convert_to_openai_function(self.add_to_context),
            self.convert_to_openai_function(self.search_web),
            self.convert_to_openai_function(self.play_youtube_video),
            self.convert_to_openai_function(self.stop_youtube_video),
            self.convert_to_openai_function(self.start_timer),
            self.convert_to_openai_function(self.stop_timer),
            self.convert_to_openai_function(self.use_camera)
        ]
        self.tools_system_message = self.open_file(tools_system_message_file)
        array_of_functions = []
        for function in self.functions:
            array_of_functions.append(self.audio_chat_functions.get_function_system_message(function.get("name")))
        self.functions_embedding = self.get_embedding(array_of_functions)

    def get_relevant_functions(self, user_message, top_k=3, threshold=0.2):
        input_embedding = self.get_embedding([user_message])
        cos_scores = util.cos_sim(input_embedding, self.functions_embedding)[0]
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        relevant_context = [self.functions[idx] for idx in top_indices if cos_scores[idx] > threshold]
        return relevant_context

    def get_embedding(self, data_array):
        return torch.tensor(self.embeddings_client.create_embeddings(data_array))

    def get_function_system_message(self, user_message=""):
        relevant_functions = self.get_relevant_functions(user_message)
        functions_system_message = ""
        for relevant_function in relevant_functions:
            function_name = relevant_function.get("name")
            functions_system_message += "\n\n" + self.audio_chat_functions.get_function_system_message(function_name)

        system_message = self.tools_system_message.replace("{functions}",
                                                           json.dumps(relevant_functions,
                                                                      indent=2)) + functions_system_message
        return system_message

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    #############
    # FUNCTIONS #
    #############

    def add_to_context(self, user_message):
        now = datetime.now()
        formatted_date = now.strftime("%Y.%B.%d_%H:%M:%S")

        with open(self.rag_vault_file, "a", encoding='utf-8') as vault_file:
            vault_file.write(f"\n{formatted_date};{user_message}")
        self.vault_updated = True
        return "Context updated with: " + str(user_message)

    def search_web(self, query):
        search = DuckDuckGoSearch(self.embeddings_client)
        results = search.duck(query, 5)
        for result in results:
            self.add_to_context(result)
        return results

    def play_youtube_video(self, query):
        search = DuckDuckGoSearch(self.embeddings_client)
        results = search.duck(query + " official youtube", 50, search_type="videos")
        if results is not None and len(results) > 0:
            try:
                title, href = search.extract_youtube_results(results, query)
                if title and href:
                    webbrowser.open(href)
                    return f"Playing {title} from YouTube: {href}"
            except TypeError:
                pass
        return None

    def stop_youtube_video(self):
        pyautogui.keyDown("command")
        pyautogui.press("w")
        time.sleep(2)
        pyautogui.keyUp("command")
        return None

    def start_timer(self, duration):
        self.timer.start(duration)

    def stop_timer(self):
        self.timer.stop()

    def use_camera(self, user_query):
        image = self.vision.capture_and_encode()
        return self.vision.ask_llm_with_image(image, user_query)

    def convert_to_openai_function(self, func):
        return {
            "name": func.__name__,
            "parameters": {
                "type": "object"
            },
        }

    def parse_function_call(self, input_str):
        try:
            match = re.search(r'<functioncall>(.*?)</functioncall>', input_str, re.DOTALL)
            if match:
                function_text = match.group(1)
                function_text = function_text.replace("{{", "{").replace("}}", "}")
                try:
                    return json.loads(function_text.strip())
                except json.JSONDecodeError:
                    return None
        except TypeError:
            pass
        return None
