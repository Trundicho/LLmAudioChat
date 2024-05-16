import base64

import cv2
from openai import OpenAI

from src.audio_chat_config import AudioChatConfig


class LlmVisionOpenAi:

    def __init__(self):
        config = AudioChatConfig().get_config()
        self.llavaClient = OpenAI(base_url=config["API_ENDPOINTS"]["AI_VISION"], api_key="not-needed")
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

    def answer_question_with_vision(self, base64_string, user_question):
        completion = self.llavaClient.chat.completions.create(
            model="local-model",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You only answer in german. You answer in short sentences and just answer the "
                                    "user's question and nothing else."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Bitte antworte nur auf deutsch und beantworte "
                                    f"nur die folgende Frage des Benutzers so kurz wie möglich: {user_question}"
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
