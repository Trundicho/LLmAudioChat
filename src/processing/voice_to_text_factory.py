from src.processing.voice_to_text import VoiceToTextWhisper
from src.processing.voice_to_text_faster import VoiceToTextFaster
from src.processing.voice_to_text_mlx import VoiceToTextMlx
from src.tools.search.search import AudioChatConfig


class VoiceToTextFactory:
    def __init__(self):
        self.voice_to_text_type = AudioChatConfig().get_config()["VOICE"]["VOICE_TO_TEXT_TYPE"]

    def create_voice_to_text(self):
        if "WHISPER" == self.voice_to_text_type:
            return VoiceToTextWhisper()
        elif "FASTER_WHISPER" == self.voice_to_text_type:
            return VoiceToTextFaster()
        elif "MLX_WHISPER" == self.voice_to_text_type:
            return VoiceToTextMlx()
        else:
            print("Voice type not supported " + self.voice_to_text_type)
            return None
