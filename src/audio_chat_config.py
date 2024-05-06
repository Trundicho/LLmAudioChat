import toml
import os


class AudioChatConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # If the config file doesn't exist, copy from the sample
        if not os.path.exists("./config.toml"):
            with open("./sample.config.toml", "r") as f_in, open("./config.toml", "w") as f_out:
                f_out.write(f_in.read())

        self.config = toml.load("./config.toml")

    def get_config(self):
        return self.config
