from threading import Thread

import vertexai
from vertexai.generative_models import GenerativeModel
import asyncio


class Gemini:
    def __init__(self):
        print("init gemini")
        vertexai.init(project="grain-defense-36857560", location="us-central1") #us-central1 europe-west3
        self.model = GenerativeModel(
            "gemini-1.0-pro-001", #claude-3-haiku@20240307, gemini-1.0-pro-001
        )

    def inference(self, model_id: str, prompt: str) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.inference2(model_id, prompt))

    async def inference2(self, model_id: str, prompt: str) -> str:
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 1,
        }

        print("=============prompt to gemini: " + prompt)
        try:
            response = self.model.generate_content(prompt, generation_config=generation_config)
            print("=============response from gemini: " + response.text)
            return response.text
        except Exception as e:
            # Handle the exception here
            print("An error occurred:", str(e))


if __name__ == '__main__':
    gemini = Gemini()
    with open('prompt.txt', 'r') as file:
        prompt_content = file.read()

    # Call the inference method with a model ID and prompt
    thread = Thread(target=lambda: gemini.inference("gemini-1.0-pro-001", prompt_content))
    thread.start()
    # response = gemini.inference("gemini-1.0-pro-001", prompt_content)

