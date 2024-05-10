import asyncio
import copy
from threading import Thread

import vertexai
from vertexai.generative_models import GenerativeModel

from src.ai.clients.ai_client import IOpenAiClient


class GeminiClient(IOpenAiClient):
    def __init__(self, tts=None):
        print("init gemini")
        vertexai.init(project="grain-defense-36857560", location="us-central1")  # us-central1 europe-west3
        self.tts = tts
        self.system_prompt = None
        self.model = None

    def ask_ai_stream(self, messages):
        messages_copy = copy.deepcopy(messages)
        for m in messages_copy:
            if "system" == m["role"]:
                if self.system_prompt is None:
                    self.system_prompt = m["content"]
                messages_copy.pop(messages_copy.index(m))
            elif "assistant" == m["role"]:
                index = messages_copy.index(m)
                messages_copy.pop(index)
                messages_copy.insert(index, {"role": "model", "content": m["content"]})
                break
        if self.model is None:
            self.model = GenerativeModel("gemini-1.5-pro-preview-0409", system_instruction=self.system_prompt)
        # gemini-1.5-pro-preview-0409
        # gemini-1.0-pro-001

        answer = ""
        answer_for_audio = ""
        is_function_call = False
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 1,
        }
        try:
            responses = self.model.generate_content(
                contents=self.convert_content_dicts_to_contents(messages_copy), generation_config=generation_config,
                stream=True)
            for response in responses:
                parts = response.candidates.__getitem__(0).content.parts
                if len(parts) > 0:
                    content = parts.pop(0).text
                    if (content is not None
                            and (content.strip().__contains__("<function")
                                 or content.strip().__contains__("functioncall"))):
                        is_function_call = True
                        print("Function call detected")
                    if content is not None:
                        content = content.replace("\n", " ").strip()
                        if content != "":
                            answer += content
                            answer_for_audio += content
                            if (answer_for_audio.__contains__(".")
                                    or answer_for_audio.__contains__("?") or answer_for_audio.__contains__("!")):
                                print(f"AI: {answer_for_audio}")
                                if not is_function_call:
                                    self.add_to_tts_queue(answer_for_audio)
                                answer_for_audio = ""
            if answer_for_audio is not None:
                print(f"AI: {answer_for_audio}")
                if not is_function_call:
                    self.add_to_tts_queue(answer_for_audio)
            return answer
        except Exception as e:
            # Handle the exception here
            print("An error occurred:", str(e))

    def my_chat(self, messages, functions):
        return self.inference2(messages)

    def inference(self, prompt: str) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.inference2(prompt))

    async def inference2(self, prompt) -> str:
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 1,
        }

        print("=============prompt to gemini: " + prompt)
        try:
            response = self.model.generate_content(contents=prompt, generation_config=generation_config)
            print("=============response from gemini: " + response.text)
            return response.text
        except Exception as e:
            # Handle the exception here
            print("An error occurred:", str(e))

    def add_to_tts_queue(self, answer_for_audio):
        if self.tts is not None:
            self.tts.add_to_queue(answer_for_audio)

    def convert_content_dicts_to_contents(self, messages):
        contents = []
        for content_dict in messages:
            content = {
                "role": content_dict["role"],
                "parts": [
                    {
                        "text": content_dict["content"],
                    }
                ],
            }
            contents.append(content)

        return contents


if __name__ == '__main__':
    gemini = GeminiClient()
    with open('prompt.txt', 'r') as file:
        prompt_content = file.read()

    # Call the inference method with a model ID and prompt
    thread = Thread(target=lambda: gemini.inference("gemini-1.0-pro-001", prompt_content))
    thread.start()
    # response = gemini.inference("gemini-1.0-pro-001", prompt_content)
