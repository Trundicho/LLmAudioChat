from src.ai.clients.claude_vertex_client import ClaudeVertex
from src.ai.clients.embeddings_client import EmbeddingsClient
from src.ai.clients.gemini_client import GeminiClient
from src.ai.clients.open_ai_client import OpenAiClient
from src.tools.search.search import AudioChatConfig


class AiClientFactory:
    def __init__(self):
        self.config = AudioChatConfig().get_config()

    def create_ai_client(self, tts):
        ai_type = self.config["AI_CONFIG"]["AI_TO_USE"]
        if "OPENAI" == ai_type:
            return OpenAiClient(tts)
        elif "CLAUDE" == ai_type:
            return ClaudeVertex(tts)
        elif "GEMINI" == ai_type:
            return GeminiClient(tts)
        else:
            print("AI type not supported " + ai_type)
            return None

    def create_embeddings_client(self):
        return EmbeddingsClient()
