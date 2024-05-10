import subprocess
import time
import queue


class TextToSpeech:
    def __init__(self):
        self.talking_queue = queue.Queue()

    def text_to_speech(self, text):
        if text is not None and text.strip() != "":
            timeout = max(5, len(text.split(" ")))
            try:
                subprocess.run(['say', text], shell=False, check=False, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"Timeout error while speaking. {timeout} seconds passed.")
                pass

    def talking_worker(self):
        while True:
            time.sleep(0.01)
            if not self.talking_queue.empty():
                task = self.talking_queue.get()
                task()
                self.talking_queue.task_done()

    def stop_talking(self):
        if not self.talking_queue.empty():
            self.talking_queue.queue.clear()
            print(f"Stopped talking")

    def add_to_queue(self, new_text):
        self.talking_queue.put(lambda text=new_text: self.text_to_speech(text))
