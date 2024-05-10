import json
import os
import re
import time
import webbrowser
from datetime import datetime

import pyautogui
import torch
from sentence_transformers import util

from src.ai.clients.ai_client_factory import AiClientFactory
from src.tools.search.search import DuckDuckGoSearch
from src.tools.timer.timer import Timer


class RagSystem:
    def __init__(self, rag_vault_file="vault.txt", tts=None):
        self.rag_vault_file = rag_vault_file
        self.conversation_history = []
        self.vault_updated = True
        self.vault_content = []
        self.vault_embeddings = torch.tensor([])
        self.ai_client = AiClientFactory().create_ai_client(tts)
        self.embeddings_client = AiClientFactory().create_embeddings_client()
        self.timer = Timer()
        self.functions = [
            self.convert_to_openai_function(self.add_to_context),
            self.convert_to_openai_function(self.search_web),
            self.convert_to_openai_function(self.play_youtube_video),
            self.convert_to_openai_function(self.stop_youtube_video),
            self.convert_to_openai_function(self.start_timer),
            self.convert_to_openai_function(self.stop_timer)
        ]

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def get_embedding(self, data_array):
        return torch.tensor(self.embeddings_client.create_embeddings(data_array))

    def get_relevant_context(self, user_input, top_k=3, threshold=0.5):
        vault_content, vault_embeddings = self.get_vault_embeddings()
        if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
            return []
        input_embedding = self.get_embedding([user_input])
        cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        relevant_context = [vault_content[idx].strip() for idx in top_indices if cos_scores[idx] > threshold]
        return relevant_context

    def get_vault_embeddings(self):
        if not self.vault_updated:
            return self.vault_content, self.vault_embeddings
        self.vault_updated = False
        if os.path.exists(self.rag_vault_file):
            with open(self.rag_vault_file, "r", encoding='utf-8') as vault_file:
                self.vault_content = vault_file.readlines()
        self.vault_embeddings = self.get_embedding(self.vault_content) if self.vault_content else []
        return self.vault_content, self.vault_embeddings

    # Check context: What's the name of my pet?
    # Check context: What's my favorite place?
    # Check context: What do I like to eat the most?

    def check_context(self, messages):
        user_message = messages[len(messages) - 1]["content"]
        relevant_context = self.get_relevant_context(user_message, 3)
        if relevant_context:
            context_str = "\n\n".join(relevant_context)
            result = (f"Possibly relevant information to answer the user's question: \n{context_str}\n\nIf there is "
                      f"relevant information, please just answer the user's question: {user_message}")
            messages[len(messages) - 1] = {"role": "user", "content": result}
            result2 = self.ai_client.ask_ai_stream(messages)  # Pass a List of messages
            return result2
        else:
            result2 = self.ai_client.ask_ai_stream(messages)
        return result2

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
        match = re.search(r'<functioncall>(.*?)</functioncall>', input_str, re.DOTALL)
        if match:
            function_text = match.group(1)
            function_text = function_text.replace("{{", "{").replace("}}", "}")
            try:
                return json.loads(function_text.strip())
            except json.JSONDecodeError:
                return None
        return None

    def get_function_system_message(self, file_name="functions_system_message.txt"):
        return self.open_file(file_name).replace("{functions}", json.dumps(self.functions, indent=2))

    def chat(self, messages):
        if messages[0]["role"] != "system":
            system_message = (f"You are an AI Agent that is an expert in following instructions. "
                              f"You will engage in conversation and provide relevant information to the "
                              f"best of your knowledge. ")
            system_message += self.get_function_system_message()
            messages.insert(0, {"role": "system", "content": system_message})
        message_content = self.check_context(messages)
        function_call = self.parse_function_call(message_content)
        if function_call:
            return self.execute_function_call(function_call)
        return message_content

    def execute_function_call(self, function_call):
        function_name = function_call["name"]
        function_arguments = {}
        if "arguments" in function_call:
            function_arguments = function_call["arguments"]
        if function_name == "search_web":
            search_result = self.search_web(**function_arguments)
            print(f"Added {len(search_result)} top search results to the context.")
            self.conversation_history.append(
                {"role": "user", "content": "Web search results: " + function_arguments["query"]})
            resp = self.ai_client.my_chat(self.conversation_history, [])
            return resp
        elif function_name == "play_youtube_video":
            played_youtube_video = self.play_youtube_video(**function_arguments)
            return f"The following video has been played: {played_youtube_video}."
        elif function_name == "stop_youtube_video":
            self.stop_youtube_video()
            return f"The youtube video has been stopped."
        elif function_name == "add_to_context":
            vault_result = self.add_to_context(**function_arguments)
            return vault_result
        elif function_name == "start_timer":
            self.start_timer(**function_arguments)
            return f"The timer has been started."
        elif function_name == "stop_timer":
            self.stop_timer()
            return f"The timer has been stopped."
        else:
            return f"Unknown function: {function_name}"


if __name__ == '__main__':
    ragSystem = RagSystem("vault.txt")
    while True:
        user_input = input("User: ")
        ragSystem.conversation_history.append({"role": "user", "content": user_input})
        response = ragSystem.chat(ragSystem.conversation_history)
        ragSystem.conversation_history.append({"role": "assistant", "content": response})
        print("Assistant:", response)
