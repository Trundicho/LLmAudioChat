from threading import Thread
from openai import OpenAI
from anthropic import AnthropicVertex
import copy

from src.ai.clients.ai_client import IOpenAiClient
from src.tools.search.search import AudioChatConfig


class ClaudeVertex(IOpenAiClient):
    def __init__(self, tts=None):
        print("init claude")
        config = AudioChatConfig().get_config()
        self.claude_model_id = config["VERTEXAI"]["CLAUDE_MODEL_ID"]
        self.client = AnthropicVertex(region=config["VERTEXAI"]["REGION"], project_id=config["VERTEXAI"][
            "PROJECTID"])

        self.embedding_model = config["API_MODELS"]["EMBEDDING"]
        self.llm_api_key = config["API_KEYS"]["OPENAI"]
        self.embedding_client = OpenAI(api_key=self.llm_api_key, base_url=config["API_ENDPOINTS"]["EMBEDDING"])
        self.tts = tts

    def my_chat(self, messages, functions):
        system_prompt = "You are a helpful assistant."
        messages_copy = copy.deepcopy(messages)
        for m in messages_copy:
            if "system" == m["role"]:
                system_prompt = m["content"]
                messages_copy.pop(messages_copy.index(m))

        try:
            message = self.client.messages.create(
                max_tokens=4096,
                messages=messages_copy,
                model=self.claude_model_id,
                system=system_prompt
            )
            return message.content[0].text
        except Exception as e:
            # Handle the exception here
            print("An error occurred:", str(e))
            return "An error occured while processing your AI request. Please try again later."

    def ask_ai_stream(self, messages):
        system_prompt = "You are a helpful assistant."
        messages_copy = copy.deepcopy(messages)
        for m in messages_copy:
            if "system" == m["role"]:
                system_prompt = m["content"]
                messages_copy.pop(messages_copy.index(m))
        answer = ""
        answer_for_audio = ""
        is_function_call = False
        try:
            with self.client.messages.stream(
                max_tokens=4096,
                messages=messages_copy,
                model=self.claude_model_id,
                system=system_prompt
            ) as stream:
                for answer_part in stream.text_stream:
                    content = answer_part
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
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "An error occured while processing your AI request. Please try again later."

    def inference(self, prompt: str) -> str:
        print("=============prompt to claude:\n\n" + prompt)
        try:
            message = self.client.messages.create(
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.claude_model_id,
            )
            print("=============response from claude:\n\n" + message.content[0].text)
            return message.content[0].text
        except Exception as e:
            # Handle the exception here
            print("An error occurred:", str(e))

    def add_to_tts_queue(self, answer_for_audio):
        if self.tts is not None:
            self.tts.add_to_queue(answer_for_audio)


if __name__ == '__main__':
    claudeVertex = ClaudeVertex()
    # with open('../rag/self_evaluations.txt', 'r') as file:
    #     prompt_content = file.read()

    # Call the inference method with a model ID and prompt
    # efforts__ = "The following is my (Angelo Romito) self evaluation over time " \
    #             "as a senior software engineer and later as a " \
    #             "staff software engineer at the company GoTo. " \
    #             "For my certificate of employment, can you please " \
    #             "distill 6 highlights from that evaluations in an abstract " \
    #             "manner so even people from other companies can understand what I did " \
    #             "and what my strengths are? Definitely highlight some of my automation efforts.\n\n"

    # content = efforts__ + prompt_content
    # thread = Thread(target=lambda:
    # print(claudeVertex.inference(model_id,
    #                        "Why is the wind blowing on earth?")))
    messages = [{"role": "user", "content": "Why is the wind blowing on earth?"}]
    thread = Thread(target=lambda:
    print(claudeVertex.ask_ai_stream(messages)))
    thread.start()
