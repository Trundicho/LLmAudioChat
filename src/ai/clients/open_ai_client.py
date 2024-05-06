from openai import OpenAI

from src.ai.clients.ai_client import IOpenAiClient
from src.search.search import AudioChatConfig


class OpenAiClient(IOpenAiClient):
    def __init__(self, tts):
        config = AudioChatConfig().get_config()
        self.open_ai_server = config["API_ENDPOINTS"]["OPENAI"]
        self.llm_api_key = config["API_KEYS"]["OPENAI"]
        self.llm_api_key = "not needed for a local LLM server"
        self.llm_model = config["API_MODELS"]["OPENAI"]
        self.client = OpenAI(api_key=self.llm_api_key, base_url=self.open_ai_server)
        self.tts = tts

    def my_chat(self, messages, functions):
        return self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0,
            top_p=0.95,
            stream=False,
            functions=functions
        ).choices[0].message.content

    def ask_ai_stream(self, messages):
        answer = ""
        answer_for_audio = ""
        is_function_call = False
        for answer_part in self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0,
                top_p=0.95,
                stream=True
        ):
            content = answer_part.choices[0].delta.content
            if content is not None and content.strip().startswith("<function"):
                is_function_call = True
                print("Function call detected")
            if content is not None:
                answer += content
                answer_for_audio += content
                if (answer_for_audio.endswith(".") and not answer_for_audio[-2].isdigit()
                        or answer_for_audio.endswith("?"
                                                     or
                                                     answer_for_audio.endswith("!"))):
                    print(f"AI: {answer_for_audio}")
                    if not is_function_call:
                        self.add_to_tts_queue(answer_for_audio)
                    answer_for_audio = ""
            else:
                print(f"AI: {answer_for_audio}")
                if not is_function_call:
                    self.add_to_tts_queue(answer_for_audio)
        return answer

    def add_to_tts_queue(self, answer_for_audio):
        if self.tts is not None:
            self.tts.add_to_queue(answer_for_audio)
