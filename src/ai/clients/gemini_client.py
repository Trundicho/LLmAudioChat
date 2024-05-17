import asyncio
import base64
import copy
from threading import Thread

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from src.ai.clients.ai_client import IOpenAiClient
from src.audio_chat_config import AudioChatConfig


class GeminiClient(IOpenAiClient):
    def __init__(self, tts=None):
        print("init gemini")
        config = AudioChatConfig().get_config()
        vertexai.init(project=config["VERTEXAI"]["PROJECTID"], location=config["VERTEXAI"]["REGION"])
        self.tts = tts
        self.system_prompt = None
        self.model = None
        self.model_id = config["VERTEXAI"]["GEMINI_MODEL_ID"]
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 1,
        }

    def ask_ai_stream_with_image(self, user_query, image):
        if self.model is None:
            self.model = GenerativeModel(self.model_id, system_instruction="Du bist ein hilfreicher Assistent der "
                                                                           "Fotos analysiert und dabei kurz die "
                                                                           "Fragen des Benutzers auf deutsch "
                                                                           "beantwortet.")
        image1 = Part.from_data(mime_type="image/png", data=base64.b64decode(image))
        responses = self.model.generate_content(
            [image1, user_query],
            generation_config=self.generation_config,
            stream=True,
        )
        return self.handle_responses(responses)


    def ask_ai_stream(self, messages):
        messages_copy = copy.deepcopy(messages)
        for m in messages_copy:
            if "system" == m["role"]:
                self.system_prompt = m["content"]
                messages_copy.pop(messages_copy.index(m))
            elif "assistant" == m["role"]:
                index = messages_copy.index(m)
                messages_copy.pop(index)
                messages_copy.insert(index, {"role": "model", "content": m["content"]})
                break
        self.model = GenerativeModel(self.model_id, system_instruction=self.system_prompt)

        try:
            responses = self.model.generate_content(
                contents=self.convert_content_dicts_to_contents(messages_copy),
                generation_config=self.generation_config,
                stream=True)
            return self.handle_responses(responses)
        except Exception as e:
            # Handle the exception here
            print("An error occurred:", str(e))

    def handle_responses(self, responses):
        answer = ""
        answer_for_audio = ""
        is_function_call = False
        for response in responses:
            parts = response.candidates.__getitem__(0).content.parts
            if len(parts) > 0:
                content = parts.pop(0).text
                if (is_function_call == False and content is not None
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

    def my_chat(self, messages, functions):
        return self.inference2(messages[len(messages) - 1]["content"])

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
    thread = Thread(target=lambda: gemini.inference(prompt_content))
    thread.start()
    # response = gemini.inference("gemini-1.0-pro-001", prompt_content)
