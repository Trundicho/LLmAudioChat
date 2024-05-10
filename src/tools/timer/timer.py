import threading
import time
import playsound


class Timer:
    def __init__(self):
        self.start_time = None
        self.duration = None
        self.is_running = False
        self.thread = threading.Thread(target=self.check)
        self.thread.start()

    def start(self, duration):
        if self.is_running:
            print("Timer is already running.")
            return

        self.start_time = time.time()
        self.duration = int(duration)
        self.is_running = True

    def stop(self):
        if not self.is_running:
            print("Timer is not running.")
            return
        self.is_running = False

    def check(self):
        while True:
            if self.is_running:
                elapsed_time = time.time() - self.start_time

                if elapsed_time >= self.duration:
                    playsound.playsound("./src/tools/timer/beep-07a.wav")
                    time.sleep(2)

                if elapsed_time >= 20:
                    self.stop()
                    return

                time.sleep(0.1)
