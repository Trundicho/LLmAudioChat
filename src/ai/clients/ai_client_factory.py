from src.ai.clients.claude_vertex_client import ClaudeVertex
from src.ai.clients.open_ai_client import OpenAiClient
from src.search.search import AudioChatConfig


class AiClientFactory():
    def __init__(self):
        self.config = AudioChatConfig().get_config()

    def create_ai_client(self, tts):
        ai_type = self.config["AI_TYPE"]["AI_TO_USE"]
        if "OPENAI" == ai_type:
            return OpenAiClient(tts)
        elif "CLAUDE" == ai_type:
            return ClaudeVertex(tts)
        else:
            print("AI type not supported " + ai_type)
            return None
