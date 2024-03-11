#!pip install torch
#!pip install noisereduce
#!pip install scipy

import requests
import base64
import numpy as np
from scipy.io.wavfile import read, write
#import noisereduce as nr
import nltk
import struct
test=False
# Define sentence split length
SENTENCE_SPLIT_LENGTH = 400

##["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","ja"]
def detect_language(sentence):
    url = "https://ruslanmv-hf-llm-api-collection.hf.space/detect"
    data = {"input_text": sentence}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        try:
            response_json = response.json()
            language = response_json.get("lang")  # Assuming "lang" is the key
            return language
        except JSONDecodeError:
            print("Error: Invalid JSON response from the language detection API.")
    else:
        print(f"Error: Language detection API call failed with status code {response.status_code}")

    return None  # Fallback if API calls fail

def split_sentences(text, max_len):
    # Apply custom rules to enforce sentence breaks with double punctuation
    text = re.sub(r"(\s*\.{2})\s*", r".\1 ", text)  # for '..'
    text = re.sub(r"(\s*\!{2})\s*", r"!\1 ", text)  # for '!!'

    # Use NLTK to split into sentences
    sentences = nltk.sent_tokenize(text)

    # Then check if each sentence is greater than max_len, if so, use textwrap to split it
    sentence_list = []
    for sent in sentences:
        if len(sent) > max_len:
            wrapped = textwrap.wrap(sent, max_len, break_long_words=True)
            sentence_list.extend(wrapped)
        else:
            sentence_list.append(sent)
    
    return sentence_list


def get_voice_streaming2(sentence, language):
    """Makes a POST request to the text-to-speech API and yields audio chunks."""
    url = "https://ruslanmv-hf-llm-api-collection.hf.space/tts"
    data = {"input_text": sentence, "from_language": language}
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    return response


def pcm_to_wav2(pcm_data, sample_rate=24000, channels=1, bit_depth=16):
    if pcm_data.startswith(b"RIFF"):
        return pcm_data

    fmt_subchunk_size = 16
    data_subchunk_size = len(pcm_data)
    chunk_size = 4 + (8 + fmt_subchunk_size) + (8 + data_subchunk_size)

    wav_header = struct.pack('<4sI4s', b'RIFF', chunk_size, b'WAVE')
    fmt_subchunk = struct.pack('<4sIHHIIHH',
                               b'fmt ', fmt_subchunk_size, 1, channels,
                               sample_rate, sample_rate * channels * bit_depth // 8,
                               channels * bit_depth // 8, bit_depth)

    data_subchunk = struct.pack('<4sI', b'data', data_subchunk_size)
    return wav_header + fmt_subchunk + data_subchunk + pcm_data

import base64
import re
def generate_speech_from_history2(history, chatbot_role, sentence):
    """
    Generates speech audio from a given sentence, performing necessary preprocessing.

    Args:
        history (list): Conversation history.
        chatbot_role (str): Role of the chatbot.
        sentence (str): The sentence to be converted to speech.

    Returns:
        list: A list of dictionaries containing text and audio (base64 encoded) for each sentence fragment.
    """
    language = "autodetect"
    if len(sentence) == 0:
        print("EMPTY SENTENCE")
        return
    # Preprocessing steps:
    # - Remove special prompt token (</s>)
    sentence = sentence.replace("</s>", "")
    # - Remove code sections (enclosed in triple backticks)
    sentence = re.sub("`.*`", "", sentence, flags=re.DOTALL)
    # - Remove inline code fragments (backticks)
    sentence = re.sub("`.*`", "", sentence, flags=re.DOTALL)
    # - Remove content within parentheses
    sentence = re.sub("\(.*\)", "", sentence, flags=re.DOTALL)
    # - Remove remaining triple backticks
    sentence = sentence.replace("```", "")
    # - Replace ellipses with spaces
    sentence = sentence.replace("...", " ")
    # - Replace parentheses with spaces
    sentence = sentence.replace("(", " ")
    sentence = sentence.replace(")", " ")
    # - Remove assistant tag
    sentence = sentence.replace("<|assistant|>","")
    if len(sentence) == 0:
        print("EMPTY SENTENCE after processing")
        return
    # - Handle punctuation at the end of sentences
    sentence = re.sub("([^\x00-\x7F]|\w)([\.。?!]+)", r"\1 \2", sentence)
    print("Sentence for speech:", sentence)
    results = []

    try:
        if len(sentence) < SENTENCE_SPLIT_LENGTH:
            sentence_list = [sentence]
        else:
            # Split longer sentences (implement your preferred split method)
            sentence_list = split_sentences(sentence, SENTENCE_SPLIT_LENGTH)
            print("detected sentences:", sentence_list)

        for sentence in sentence_list:
            print("- sentence =", sentence)
            if any(c.isalnum() for c in sentence):
                if language == "autodetect":
                    language = detect_language(sentence)  # Detect language on first call
                    print("language",language)
                audio_stream = get_voice_streaming2(sentence, language)
                if audio_stream is not None:
                    sentence_wav_bytestream = b""
                    # Process audio chunks
                    for chunk in audio_stream:
                        if chunk is not None:
                            sentence_wav_bytestream += chunk
                    # Encode WAV to base64
                    base64_audio = base64.b64encode(pcm_to_wav2(sentence_wav_bytestream)).decode('utf8')
                    print("base64_audio",base64_audio[:10])
                    results.append({ "text": sentence, "audio": base64_audio })
                else:
                    # Handle the case where the audio stream is None (e.g., silent response)
                    results.append({ "text": sentence, "audio": "" })

    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything , need to restart
            print(
                f"Exit due to: Unrecoverable exception caused by prompt:{sentence}",
                flush=True,
            )
            #This error is unrecoverable need to restart space
            #api.restart_space(repo_id=repo_id)
        else:
            print("RuntimeError: non device-side assert error:", str(e))
            raise e

    return results

if test:
    # Example usage
    history = []
    chatbot_role = "assistant"
    sentence = "Hello, how can I help you?"
    result = generate_speech_from_history2(history, chatbot_role, sentence)
    print(result)