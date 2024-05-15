from abc import abstractmethod, ABC


class IVoiceToText(ABC):

    @abstractmethod
    def voice_to_text(self, source, language, whisper_model_type: str = "base"):
        pass
