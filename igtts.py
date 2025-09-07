import argparse
import os
import wave

from google import genai
from google.genai import types

# 定义常量以提高代码可读性和可维护性
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Zephyr"
DEFAULT_INPUT_FILE = "input.txt"
DEFAULT_OUTPUT_FILE = "output.wav"
WAV_CHANNELS = 1
WAV_RATE = 24000
WAV_SAMPLE_WIDTH = 2


def save_as_wav_file(filename: str, pcm_data: bytes):
    """
    将 PCM 数据保存为 WAV 文件。
    """
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(WAV_CHANNELS)
            wf.setsampwidth(WAV_SAMPLE_WIDTH)
            wf.setframerate(WAV_RATE)
            wf.writeframes(pcm_data)
        print(f"音频已成功保存到 '{filename}'")
    except Exception as e:
        print(f"错误：保存 WAV 文件失败 - {e}")


def gemini_tts(text: str, voice: str, file_name: str):
    """
    使用 Gemini TTS 模型将文本转换为语音并保存为 WAV 文件。

    Args:
        text: 要转换成音频的文本字符串。
        voice: 语音名称 (例如: 'Kore')。
        file_name: 保存音频文件的文件名 (例如: 'output.wav')。
    """
    print("开始生成音频...")
    client = genai.Client()
    try:
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice,
                        )
                    )
                ),
            )
        )
        # 检查响应中是否有音频数据
        if response.candidates and response.candidates[0].content.parts and \
                response.candidates[0].content.parts[0].inline_data:
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            save_as_wav_file(file_name, pcm_data)
        else:
            print("错误：Gemini API 响应中没有找到音频数据。")
            print("响应详情：", response)
    except Exception as e:
        print(f"错误：Gemini API 调用失败 - {e}")


def main():
    parser = argparse.ArgumentParser(description="Gemini 文本转语音（TTS）命令行工具。")
    parser.add_argument("text", nargs="?", help="指定语音文本的内容。如果未提供，请使用 --input-file。")
    parser.add_argument("-i", "--input-file", default=DEFAULT_INPUT_FILE, help="指定语音文本的文件。")
    parser.add_argument("-l", "--list-voices", action="store_true", help="列出语音的语音名称。")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, help=f"指定语音的语音名称（默认：{DEFAULT_VOICE}）。")
    parser.add_argument("-o", "--output-file", default=DEFAULT_OUTPUT_FILE,
                        help=f"指定音频保存的文件（默认：{DEFAULT_OUTPUT_FILE}）。")

    args = parser.parse_args()

    if args.list_voices:
        voices = ["Zephyr", "Puck", "Erinome", "Kore", "Gacrux", "Autonoe", "Iapetus"]
        print("语音名称的列表:")
        for voice in voices:
            print(voice)
        print("……")
        return

    # 处理输入文本
    text_to_speak = ""
    if args.text:
        text_to_speak = args.text
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"错误：找不到文件 '{args.input_file}'")
            return
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                text_to_speak = f.read()
        except IOError as e:
            print(f"错误：读取文件 '{args.input_file}' 失败 - {e}")
            return

    if not text_to_speak.strip():
        print("错误：没有要朗读的文本。")
        return

    print(f"语音内容: {text_to_speak[:30]}{'...' if len(text_to_speak) > 30 else ''}")
    print(f"语音名称: {args.voice}")
    gemini_tts(text_to_speak, args.voice, args.output_file)

if __name__ == "__main__":
    main()
