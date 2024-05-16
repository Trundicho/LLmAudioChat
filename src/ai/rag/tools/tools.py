import json
import re
import time
import webbrowser
from datetime import datetime

import pyautogui

from src.ai.clients.ai_client_factory import AiClientFactory
from src.ai.clients.llm_vision import LlmVisionOpenAi
from src.tools.search.search import DuckDuckGoSearch
from src.tools.timer.timer import Timer


class Tools:
    def __init__(self, rag_vault_file="vault.txt"):
        self.rag_vault_file = rag_vault_file
        self.vault_updated = False
        self.embeddings_client = AiClientFactory().create_embeddings_client()
        self.timer = Timer()
        self.vision = LlmVisionOpenAi()
        self.functions = [
            self.convert_to_openai_function(self.add_to_context),
            self.convert_to_openai_function(self.search_web),
            self.convert_to_openai_function(self.play_youtube_video),
            self.convert_to_openai_function(self.stop_youtube_video),
            self.convert_to_openai_function(self.start_timer),
            self.convert_to_openai_function(self.stop_timer),
            self.convert_to_openai_function(self.answer_question_with_vision)
        ]

    def get_relevant_functions(self, user_message):
        return self.functions

    def get_function_system_message(self, file_name="tools_system_message.txt"):
        return self.open_file(file_name).replace("{functions}", json.dumps(self.functions, indent=2))

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

    #     Add to context: I have a son and a daughter

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

    def answer_question_with_vision(self, user_query):
        image = self.vision.capture_and_encode()
        return self.vision.answer_question_with_vision(image, user_query)

    def convert_to_openai_function(self, func):
        return {
            "name": func.__name__,
            "description": func.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Email address of the recipient"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject of the email"
                    },
                    "body": {
                        "type": "string",
                        "description": "Body of the email"
                    },
                    "attachment": {
                        "type": "string",
                        "description": "Path to an attachment"
                    },
                    "query": {
                        "type": "string",
                        "description": "Query to search on Google"
                    },
                    "user_message": {
                        "type": "string",
                        "description": "Users provided message to compare to the context"
                    },
                },
                "required": ["recipient", "subject", "body"],
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
