import argparse
import os
import wave
from google import genai
from google.genai import types

# =====================
# 常量定义
# =====================
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Zephyr"
DEFAULT_INPUT_FILE = "input.txt"
DEFAULT_OUTPUT_FILE = "output.wav"
WAV_CHANNELS = 1
WAV_RATE = 24000
WAV_SAMPLE_WIDTH = 2

AVAILABLE_VOICES = [
    "Zephyr", "Puck", "Erinome", "Kore", "Gacrux", "Autonoe", "Iapetus"
]


# =====================
# 工具函数
# =====================
def save_as_wav(filename: str, pcm_data: bytes):
    """保存 PCM 数据为 WAV 文件。"""
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(WAV_CHANNELS)
            wf.setsampwidth(WAV_SAMPLE_WIDTH)
            wf.setframerate(WAV_RATE)
            wf.writeframes(pcm_data)
        print(f"✅ 音频已保存到：{filename}")
    except Exception as e:
        print(f"❌ 保存 WAV 文件失败：{e}")


def read_text(source_text: str | None, file_path: str) -> str:
    """优先使用命令行文本，否则从文件读取。"""
    if source_text:
        return source_text.strip()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到输入文件：{file_path}")

    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()


def gemini_tts(text: str, voice: str, output_path: str):
    """调用 Gemini 模型生成语音并保存。"""
    print("🎤 正在生成语音，请稍候…")

    client = genai.Client()
    try:
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        )

        part = (
            response.candidates[0]
            .content.parts[0]
            .inline_data
            if response.candidates
            else None
        )

        if not part or not part.data:
            raise ValueError("Gemini API 未返回有效音频数据")

        save_as_wav(output_path, part.data)

    except Exception as e:
        print(f"❌ 生成语音失败：{e}")


# =====================
# 主程序入口
# =====================
def main():
    parser = argparse.ArgumentParser(description="Gemini 文本转语音（TTS）工具")
    parser.add_argument("text", nargs="?", help="要朗读的文本（可选）")
    parser.add_argument("-i", "--input-file", default=DEFAULT_INPUT_FILE, help="指定输入文本文件")
    parser.add_argument("-o", "--output-file", default=DEFAULT_OUTPUT_FILE, help="指定输出音频文件名")
    parser.add_argument("-v", "--voice", default=DEFAULT_VOICE, help=f"语音名称（默认：{DEFAULT_VOICE}）")
    parser.add_argument("-l", "--list-voices", action="store_true", help="列出可用语音名称")

    args = parser.parse_args()

    if args.list_voices:
        print("🎙️ 可用语音名称：")
        for v in AVAILABLE_VOICES:
            print(" -", v)
        return

    try:
        text = read_text(args.text, args.input_file)
        if not text:
            raise ValueError("输入文本为空")

        print(f"📖 文本预览：{text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"🗣️ 使用语音：{args.voice}")

        gemini_tts(text, args.voice, args.output_file)

    except Exception as e:
        print(f"❌ 错误：{e}")


if __name__ == "__main__":
    main()
