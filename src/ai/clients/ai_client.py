from abc import abstractmethod, ABC


class IOpenAiClient(ABC):

    @abstractmethod
    def my_chat(self, messages, functions):
        pass

    @abstractmethod
    def create_embeddings(self, data_array):
        pass

    @abstractmethod
    def ask_ai_stream(self, messages):
        pass
