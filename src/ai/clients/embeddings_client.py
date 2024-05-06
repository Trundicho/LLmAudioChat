from openai import OpenAI

from src.audio_chat_config import AudioChatConfig


class EmbeddingsClient:

    def __init__(self):
        config = AudioChatConfig().get_config()
        self.llm_api_key = config["API_KEYS"]["OPENAI"]
        self.embedding_server = config["API_ENDPOINTS"]["EMBEDDING"]
        self.embedding_model = config["API_MODELS"]["EMBEDDING"]
        self.embedding_client = OpenAI(api_key=self.llm_api_key, base_url=self.embedding_server)

    def create_embeddings(self, data_array):
        embeddings = []
        for data_array in data_array:
            embeddings.append(self.embedding_client.embeddings.create(input=data_array,
                                                                      model=self.embedding_model).data[0].embedding)
        return embeddings
