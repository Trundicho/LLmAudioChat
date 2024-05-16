import base64

import cv2
from openai import OpenAI

from src.ai.clients.gemini_client import GeminiClient
from src.audio_chat_config import AudioChatConfig


class LlmVision:

    def __init__(self, tts=None):
        config = AudioChatConfig().get_config()
        openai_vision_endpoint = config["API_ENDPOINTS"]["AI_VISION"]
        self.ai_vision_to_use = config["AI_CONFIG"]["AI_VISION_TO_USE"]
        if self.ai_vision_to_use == "OPENAI":
            self.llavaClient = OpenAI(base_url=openai_vision_endpoint, api_key="not-needed")
        elif self.ai_vision_to_use == "GEMINI":
            self.geminiClient = GeminiClient(tts)
        self.image_path = "CatWithMouse.png"

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def capture_and_encode(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        cv2.imwrite("captured_image.png", frame)
        _, buffer = cv2.imencode('.png', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def ask_llm_with_image(self, base64_string, user_query):
        if self.ai_vision_to_use == "GEMINI":
            return self.geminiClient.ask_ai_stream_with_image(user_query, base64_string)
        elif self.ai_vision_to_use == "OPENAI":
            return self.openai_with_image(base64_string, user_query)

    def openai_with_image(self, base64_string, user_question):
        completion = self.llavaClient.chat.completions.create(
            model="local-model",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Du bist ein KI Assistent der Bilder analysiert und du antwortest "
                                    "ausschließlich auf deutsch und so kurz wie möglich."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Bitte antworte ausschließlich auf deutsch und beantworte "
                                    f"meine Frage so kurz wie möglich: {user_question}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_string}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
            stream=False
        )
        # for chunk in completion:
        #     if chunk.choices[0].delta.content:
        #         print(chunk.choices[0].delta.content, end="", flush=True)
        return completion.choices[0].message.content
