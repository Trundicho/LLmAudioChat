# Local LLM Audio Chat

This simple python script allows you to talk via microphone with a locally running LLM server via openai api.
It uses openai whisper and if all is installed and setup does not require any internet connection.

For the audio output it uses the **unix say command**.

Tested with very good results with LM Studio and TheBloke openhermes 2.5 mistral 16k 7B.

## Setup

**Disclaimer:** Developed and tested only on macos apple silicon machines.

**Requirements:**
- ```python3.11```
- ```brew remove portaudio (For macos users in case of problems)```
- ```brew install portaudio (For macos users in case of problems)```
- On a macos it's **recommended to install a high quality voice** like siri (System settings - Spoken Content).


First, install the dependencies.

```
pip install -r requirements.txt
```

OR if you want to use Apple MLX



**Configuration:**

You should **configure your language, quality and LLM settings** in the ```local_llm_audio_chat.py``` file

### Apple silicon MLX
To get a 30% performance increase using openai whisper you can use Apple MLX.

```
pip install -r requirements_mlx.txt
```

In ```local_llm_audio_chat.py``` you can configure to use Apple MLX with the flag ```use_apple_mlx```

## Run

```
python3 local_llm_audio_chat.py
```