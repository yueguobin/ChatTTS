"""
openai_api.py
This module implements a FastAPI-based text-to-speech API compatible with OpenAI's interface specification.

Main features and improvements:
- Use app.state to manage global state, ensuring thread safety
- Add exception handling and unified error responses to improve stability
- Support multiple voice options and audio formats for greater flexibility
- Add input validation to ensure the validity of request parameters
- Support additional OpenAI TTS parameters (e.g., speed) for richer functionality
- Implement health check endpoint for easy service status monitoring
- Use asyncio.Lock to manage model access, improving concurrency performance
- Load and manage speaker embedding files to support personalized speech synthesis
"""

import io
import os
import sys
import asyncio
import time
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import torch

# Cross-platform compatibility settings
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Set working directory and add to system path
now_dir = os.getcwd()
sys.path.append(now_dir)

# Import necessary modules
import ChatTTS
from tools.audio import pcm_arr_to_mp3_view, pcm_arr_to_ogg_view, pcm_arr_to_wav_view
from tools.logger import get_logger
from tools.normalizer.en import normalizer_en_nemo_text
from tools.normalizer.zh import normalizer_zh_tn

# Initialize logger
logger = get_logger("Command")

# Initialize FastAPI application
app = FastAPI()

# Voice mapping table
# Download stable voices:
# ModelScope Community: https://modelscope.cn/studios/ttwwwaa/ChatTTS_Speaker
# HuggingFace: https://huggingface.co/spaces/taa/ChatTTS_Speaker
VOICE_MAP = {
    "default": "1528.pt",
    "alloy": "1397.pt",
    "echo": "2155.pt",
}

# Supported voices (including those not in VOICE_MAP)
SUPPORTED_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}

# Allowed audio formats
ALLOWED_FORMATS = {"mp3", "wav", "ogg", "opus", "aac", "flac", "pcm"}


@app.on_event("startup")
async def startup_event():
    """Load ChatTTS model and default speaker embedding when the application starts"""
    # Initialize ChatTTS and async lock
    app.state.chat = ChatTTS.Chat(get_logger("ChatTTS"))
    app.state.model_lock = asyncio.Lock()  # Use async lock instead of thread lock

    # Register text normalizers
    app.state.chat.normalizer.register("en", normalizer_en_nemo_text())
    app.state.chat.normalizer.register("zh", normalizer_zh_tn())

    logger.info("Initializing ChatTTS...")
    if app.state.chat.load(source="huggingface"):
        logger.info("Model loaded successfully.")
    else:
        logger.error("Model loading failed, exiting application.")
        raise RuntimeError("Failed to load ChatTTS model")

    # Load default speaker embedding
    # Preload all supported speaker embeddings into memory at startup to avoid repeated loading during runtime
    app.state.spk_emb_map = {}
    for voice, spk_path in VOICE_MAP.items():
        if os.path.exists(spk_path):
            app.state.spk_emb_map[voice] = torch.load(
                spk_path, map_location=torch.device("cpu")
            )
            logger.info(f"Preloading speaker embedding: {voice} -> {spk_path}")
        else:
            logger.warning(f"Speaker embedding not found: {spk_path}, skipping preload")
    app.state.spk_emb = app.state.spk_emb_map.get("default")  # Default embedding


# Request parameter whitelist
ALLOWED_PARAMS = {
    "model",
    "input",
    "voice",
    "response_format",
    "speed",
    "stream",
    "output_format",
    "instructions",
}


class OpenAITTSRequest(BaseModel):
    """OpenAI TTS request data model"""

    model: str = Field(..., description="Speech synthesis model, supports: gpt-4o-mini-tts, tts-1, tts-1-hd")
    input: str = Field(
        ..., description="Text content to synthesize", max_length=2048
    )  # Length limit
    voice: Optional[str] = Field(
        "default", description="Voice selection, supports: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, shimmer"
    )
    response_format: Optional[str] = Field(
        "mp3", description="Audio format: mp3, wav, ogg, opus, aac, flac, pcm"
    )
    speed: Optional[float] = Field(
        1.0, ge=0.5, le=2.0, description="Speed, range 0.5-2.0"
    )
    stream: Optional[bool] = Field(False, description="Whether to stream")
    output_format: Optional[str] = "mp3"  # Optional formats: mp3, wav, ogg
    instructions: Optional[str] = Field(None, description="Voice instructions (accepted but not used in current version)")
    extra_params: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Unsupported extra parameters"
    )

    @classmethod
    def validate_request(cls, request_data: Dict):
        """Filter unsupported request parameters"""
        # Don't force model value, allow various OpenAI model names
        unsupported_params = set(request_data.keys()) - ALLOWED_PARAMS
        if unsupported_params:
            logger.warning(f"Ignoring unsupported parameters: {unsupported_params}")
        return {key: request_data[key] for key in ALLOWED_PARAMS if key in request_data}


# Unified error response
@app.exception_handler(Exception)
async def custom_exception_handler(request, exc):
    """Custom exception handler"""
    logger.error(f"Error: {str(exc)}")
    return JSONResponse(
        status_code=getattr(exc, "status_code", 500),
        content={"error": {"message": str(exc), "type": exc.__class__.__name__}},
    )


def normalize_audio_format(format_str: str) -> str:
    """将不支持的音频格式统一映射到 wav"""
    if format_str in {"opus", "aac", "flac", "pcm"}:
        return "wav"
    return format_str

@app.post("/v1/audio/speech")
async def generate_voice(request_data: Dict):
    """Handle speech synthesis request"""
    request_data = OpenAITTSRequest.validate_request(request_data)
    request = OpenAITTSRequest(**request_data)

    logger.info(
        f"Received request: text={request.input}..., voice={request.voice}, stream={request.stream}"
    )

    # Validate and normalize audio format
    if request.response_format not in ALLOWED_FORMATS:
        raise HTTPException(
            400,
            detail=f"Unsupported audio format: {request.response_format}, supported formats: {', '.join(ALLOWED_FORMATS)}",
        )
    
    # Normalize audio format (opus, aac, flac, pcm -> wav)
    normalized_format = normalize_audio_format(request.response_format)

    # Handle voice parameter with fallback logic
    if request.voice not in SUPPORTED_VOICES:
        logger.warning(f"Voice '{request.voice}' not supported, using 'default' instead")
        request.voice = "default"
    
    # Load speaker embedding for the specified voice
    spk_emb = app.state.spk_emb_map.get(request.voice, app.state.spk_emb_map.get("default"))

# 推理参数详解
# 参数名称,类型,默认值,含义及作用
# text,列表,[request.input],要合成语音的文本内容。 这是一个列表，通常包含一个或多个文本字符串。 request.input 即是用户通过 API 传入的文本。
# stream,布尔值,request.stream,是否启用流式传输。 如果为 True，音频数据将分块实时发送给客户端，以减少延迟。
# lang,字符串,None,语言设置。 指定输入文本的语言。设置为 None 可能表示模型会根据文本内容自动检测或使用默认语言。
# skip_refine_text,布尔值,True,跳过文本细化 (Text Refinement)。 如果设置为 True，模型将不会对输入文本进行额外的、基于概率的细化处理。这通常用于提高速度和确定性，但可能会牺牲一部分自然度。
# refine_text_only,布尔值,False,仅进行文本细化。 如果设置为 True，模型将只运行文本处理部分并返回细化后的文本，而不会生成音频。通常用于调试或测试文本预处理效果。
# use_decoder,布尔值,True,使用解码器进行音频生成。 如果设置为 True，模型会运行核心的声码器/解码器部分来将生成的声学特征转换为最终的波形音频。这是音频合成的必要步骤。
# audio_seed,整数,12345678,音频生成阶段的随机种子。 控制音频波形细节和随机性。设置一个固定值可以确保每次合成的音质细节和韵律具有可重复性。
# """text_seed""",整数,(被注释),文本处理阶段的随机种子。 （在您的代码中被注释掉了） 如果启用，它将控制文本细化和分词等过程的随机性，确保在文本预处理方面是可重复的。
# do_text_normalization,布尔值,True,"执行文本规范化 (Text Normalization, TN)。 TN 将文本中的数字、符号、缩写等转换为完整的、可发音的词语（例如，把 1997 转换为 一九九七年 或 一千九百九十七年）。"
# do_homophone_replacement,布尔值,True,执行同音字替换。 这是针对中文 TTS 的一个重要功能，用于处理多音字（即一个字有多个发音）。它会尝试根据上下文选择最合适的发音。
    # Inference parameters
    params_infer_main = {
        "text": [request.input],
        "stream": request.stream,
        "lang": None,
        "skip_refine_text": True,  # Do not use text refinement
        "refine_text_only": False,
        "use_decoder": True,
        "audio_seed": 12345678,
        #"text_seed": 57689395,  # Random seed for text processing, used to control text refinement
        "do_text_normalization": True,  # Perform text normalization
        "do_homophone_replacement": True,  # Perform homophone replacement
    }

    # Inference code parameters
    # Map OpenAI speed (0.5-2.0) to ChatTTS speed (1-9)
    # 0.5 → 1, 2.0 → 9, linear mapping in between
    speed_val = max(1, min(9, int(round((request.speed - 0.5) * 8 / 1.5 + 1))))
    
# 核心控制参数 (Generation Control)
# 参数名称,类型,默认值,含义
# prompt,字符串,"f""[speed_{speed_val}]""",控制推理的启动文本或风格标签。 在您的例子中，它用于通过特定标签来控制语音的语速 (speed_val 变量传入的数值)。ChatTTS 使用这类标签来引导生成风格。
# top_P,浮点数,0.7,核采样 (Nucleus Sampling) 阈值。 仅从累积概率达到 top_P 的最高概率词汇集合中进行采样。值越小，生成的语音越保守和确定；值越大，多样性和随机性越高，但可能引入更多错误。
# top_K,整数,20,Top-K 采样参数。 仅从概率最高的 K 个词汇中进行采样。它限制了采样空间，防止模型选择概率极低的词汇。
# temperature,浮点数,0.3,采样温度。 控制生成过程的随机性。值越低（如 0.3），输出越确定和保守；值越高，输出越随机和富有表现力（如 1.0）。较低的温度通常用于提高语音的稳定性和清晰度。
# repetition_penalty,浮点数,1.1,重复惩罚。 用于降低模型重复生成相同音素或序列的概率。大于 1.0 会抑制重复，小于 1.0 会鼓励重复。

# 长度与限制参数 (Length & Boundary)
#参数名称,类型,默认值,含义
#max_new_token,整数,2048,生成音频的最大长度。 这是指模型在生成过程中可以产生的最大离散 token 数量。它间接限制了最终语音的时长。
#min_new_token,整数,0,生成音频的最小长度。 设置为 0 意味着不作最小长度限制。
#ensure_non_empty,布尔值,True,确保输出非空。 如果设置为 True，它会强制模型至少生成一个有效的音频 token，以避免像您之前遇到的“zero-size array”错误。

# 音色和状态参数 (Voice & State)
#参数名称,类型,默认值,含义
#spk_emb,张量 (Tensor),spk_emb,说话人嵌入 (Speaker Embedding)。 这是一个张量，代表了用于合成语音的特定音色或说话风格。这是实现多音色或克隆音色的关键参数。
#spk_smp,任何,None,说话人采样器 (Speaker Sampler)。 用于更细致地控制说话人特征的采样逻辑。通常在高级或定制化推理中使用。
#txt_smp,任何,None,文本采样器 (Text Sampler)。 用于控制文本 token 如何被采样的逻辑。通常在高级或定制化推理中使用。
#manual_seed,整数,None,手动随机种子。 如果设置为一个整数，可以确保每次推理的随机过程（如采样）都是相同的，从而实现可重复的结果。

# 流式和性能参数 (Streaming & Performance)
#参数名称,类型,默认值,含义
#show_tqdm,布尔值,True,显示进度条。 如果设置为 True，在生成过程中会显示 tqdm 进度条（如您日志中所示的 `61%
#stream_batch,整数,24,流式传输的批量大小。 在进行音频流式输出时，这是每次传输的音频 token 块的大小。较大的值可能会提高效率，但会增加延迟。
#stream_speed,整数,9600,流式传输的最小速率（字节/秒）。 用于确保流式输出的速率。这个值可能是一个目标速率，以提供流畅的实时播放体验。
#pass_first_n_batches,整数,2,跳过前 N 个批次。 在流式传输中，模型可能需要一定量的初始计算才能开始产生有效的音频。该参数指示在开始流式传输音频块之前，要等待模型完成前 N 个批次的生成。这有助于隐藏初始延迟并确保流的起始音频质量。
    params_infer_code = app.state.chat.InferCodeParams(
        prompt=f"[speed_{speed_val}]",  # Convert to format supported by ChatTTS
        top_P=0.1,
        top_K=20,
        temperature=0.1,
        repetition_penalty=1.1,
        max_new_token=2048,
        min_new_token=0,
        show_tqdm=True,
        ensure_non_empty=True,
        manual_seed=1234567890,
        spk_emb=spk_emb,
        spk_smp=None,
        txt_smp=None,
        stream_batch=12,
        stream_speed=24000,
        pass_first_n_batches=2,
    )

    try:
        async with app.state.model_lock:
            wavs = app.state.chat.infer(
                text=params_infer_main["text"],
                stream=params_infer_main["stream"],
                lang=params_infer_main["lang"],
                skip_refine_text=params_infer_main["skip_refine_text"],
                use_decoder=params_infer_main["use_decoder"],
                do_text_normalization=params_infer_main["do_text_normalization"],
                do_homophone_replacement=params_infer_main["do_homophone_replacement"],
                # params_refine_text = params_refine_text,
                params_infer_code=params_infer_code,
            )
    except Exception as e:
        raise HTTPException(500, detail=f"Speech synthesis failed: {str(e)}")

    def generate_wav_header(sample_rate=2400, bits_per_sample=16, channels=1):
        """Generate WAV file header (without data length)"""
        header = bytearray()
        header.extend(b"RIFF")
        header.extend(b"\xff\xff\xff\xff")  # File size unknown
        header.extend(b"WAVEfmt ")
        header.extend((16).to_bytes(4, "little"))  # fmt chunk size
        header.extend((1).to_bytes(2, "little"))  # PCM format
        header.extend((channels).to_bytes(2, "little"))  # Channels
        header.extend((sample_rate).to_bytes(4, "little"))  # Sample rate
        byte_rate = sample_rate * channels * bits_per_sample // 8
        header.extend((byte_rate).to_bytes(4, "little"))  # Byte rate
        block_align = channels * bits_per_sample // 8
        header.extend((block_align).to_bytes(2, "little"))  # Block align
        header.extend((bits_per_sample).to_bytes(2, "little"))  # Bits per sample
        header.extend(b"data")
        header.extend(b"\xff\xff\xff\xff")  # Data size unknown
        return bytes(header)

    # Handle audio output format
    def convert_audio(wav, format):
        """Convert audio format"""
        if format == "mp3":
            return pcm_arr_to_mp3_view(wav)
        elif format == "wav":
            return pcm_arr_to_wav_view(
                wav, include_header=False
            )  # No header in streaming
        elif format == "ogg":
            return pcm_arr_to_ogg_view(wav)
        return pcm_arr_to_mp3_view(wav)

    # Return streaming audio data
    if request.stream:
        first_chunk = True

        async def audio_stream():
            nonlocal first_chunk
            for wav in wavs:
                if normalized_format == "wav" and first_chunk:
                    yield generate_wav_header()  # Send WAV header
                    first_chunk = False
                yield convert_audio(wav, normalized_format)

        media_type = "audio/wav" if normalized_format == "wav" else "audio/mpeg"
        return StreamingResponse(audio_stream(), media_type=media_type)

    # Return audio file directly
    if normalized_format == "wav":
        music_data = pcm_arr_to_wav_view(wavs[0])
    else:
        music_data = convert_audio(wavs[0], normalized_format)

    return StreamingResponse(
        io.BytesIO(music_data),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=output.{request.response_format}"
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": bool(app.state.chat)}
