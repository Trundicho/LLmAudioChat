class AudioChatFunctions:
    def __init__(self):
        root = "src/ai/rag/tools/"
        self.functions_map = {
            "use_screenshot_to_answer_question": self.open_file(f"{root}use_screenshot.txt"),
            "use_camera_to_answer_question": self.open_file(f"{root}use_camera.txt"),
            "add_to_context": self.open_file(f"{root}add_to_context.txt"),
            "play_youtube_video": self.open_file(f"{root}play_youtube_video.txt"),
            "stop_youtube_video": self.open_file(f"{root}stop_youtube_video.txt"),
            "search_web": self.open_file(f"{root}search_web.txt"),
            "start_timer": self.open_file(f"{root}start_timer.txt"),
            "stop_timer": self.open_file(f"{root}stop_timer.txt")
        }

    def open_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()

    def get_function_system_message(self, function_name):
        return self.functions_map.get(function_name)
