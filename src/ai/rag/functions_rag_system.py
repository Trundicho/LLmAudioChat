import os

import torch
from sentence_transformers import util

from src.ai.clients.ai_client_factory import AiClientFactory
from src.ai.clients.llm_vision import LlmVisionOpenAi
from src.ai.rag.tools.tools import Tools

from src.tools.timer.timer import Timer


class RagSystem:
    def __init__(self, rag_vault_file="vault.txt", tts=None):
        self.rag_vault_file = rag_vault_file
        self.tools = Tools(rag_vault_file)
        self.conversation_history = []
        self.vault_updated = True
        self.vault_content = []
        self.vault_embeddings = torch.tensor([])
        self.tts = tts
        self.ai_client = AiClientFactory().create_ai_client(tts)
        self.embeddings_client = AiClientFactory().create_embeddings_client()
        self.timer = Timer()
        self.vision = LlmVisionOpenAi()

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

    def chat(self, messages):
        if messages[0]["role"] != "system":
            system_message = (f"You are an AI Agent that is an expert in following instructions. "
                              f"You will engage in conversation and provide relevant information to the "
                              f"best of your knowledge. ")
            system_message += self.tools.get_function_system_message()
            messages.insert(0, {"role": "system", "content": system_message})
        message_content = self.check_context(messages)
        function_call = self.tools.parse_function_call(message_content)
        if function_call:
            return self.execute_function_call(function_call)
        return message_content

    def execute_function_call(self, function_call):
        function_name = function_call["name"]
        function_arguments = {}
        try:
            if "arguments" in function_call:
                function_arguments = function_call["arguments"]
            if function_name == "search_web":
                search_result = self.tools.search_web(**function_arguments)
                print(f"Added {len(search_result)} top search results to the context.")
                self.conversation_history.append(
                    {"role": "user", "content": "Web search results: " + function_arguments["query"]})
                resp = self.ai_client.my_chat(self.conversation_history, [])
                return resp
            elif function_name == "play_youtube_video":
                played_youtube_video = self.tools.play_youtube_video(**function_arguments)
                return f"The following video has been played: {played_youtube_video}."
            elif function_name == "stop_youtube_video":
                self.tools.stop_youtube_video()
                return f"The youtube video has been stopped."
            elif function_name == "add_to_context":
                vault_result = self.tools.add_to_context(**function_arguments)
                return vault_result
            elif function_name == "start_timer":
                self.tools.start_timer(**function_arguments)
                return f"The timer has been started."
            elif function_name == "stop_timer":
                self.tools.stop_timer()
                return f"The timer has been stopped."
            elif function_name == "answer_question_with_vision":
                answer = self.tools.answer_question_with_vision(**function_arguments)
                self.conversation_history.append(
                    {"role": "assistant", "content": answer})
                if self.tts is not None:
                    self.tts.add_to_queue(answer)
                return answer
            else:
                return f"Unknown function: {function_name}"
        except Exception as e:
            print(f"Error executing function {function_name}: {e}")
            return f"An error occurred while executing the function {function_name}."


if __name__ == '__main__':
    ragSystem = RagSystem("vault.txt")
    while True:
        user_input = input("User: ")
        ragSystem.conversation_history.append({"role": "user", "content": user_input})
        response = ragSystem.chat(ragSystem.conversation_history)
        ragSystem.conversation_history.append({"role": "assistant", "content": response})
        print("Assistant:", response)
