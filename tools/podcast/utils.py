"""
utils.py

Functions:
- generate_script: Get the dialogue from the LLM.
- call_llm: Call the LLM with the given prompt and dialogue format.
- parse_url: Parse the given URL and return the text content.
- generate_podcast_audio: Generate audio for podcast using TTS or advanced audio models.
"""

# Standard library imports
import time
from typing import Any, Union

# Third-party imports
import requests
from bark import SAMPLE_RATE, generate_audio, preload_models
from gradio_client import Client
from openai import OpenAI
from pydantic import ValidationError
from scipy.io.wavfile import write as write_wav

# Local imports
from constants import (
    FIREWORKS_API_KEY,
    FIREWORKS_BASE_URL,
    FIREWORKS_MODEL_ID,
    FIREWORKS_MAX_TOKENS,
    FIREWORKS_TEMPERATURE,
    FIREWORKS_JSON_RETRY_ATTEMPTS,
    MELO_API_NAME,
    MELO_TTS_SPACES_ID,
    MELO_RETRY_ATTEMPTS,
    MELO_RETRY_DELAY,
    JINA_READER_URL,
    JINA_RETRY_ATTEMPTS,
    JINA_RETRY_DELAY,
)
from schema import ShortDialogue, MediumDialogue

# Initialize clients
fw_client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=FIREWORKS_API_KEY)
hf_client = Client(MELO_TTS_SPACES_ID)

# Download and load all models for Bark
preload_models()


def generate_script(
    system_prompt: str,
    input_text: str,
    output_model: Union[ShortDialogue, MediumDialogue],
) -> Union[ShortDialogue, MediumDialogue]:
    """Get the dialogue from the LLM."""

    # Call the LLM
    response = call_llm(system_prompt, input_text, output_model)
    response_json = response.choices[0].message.content

    # Validate the response
    for attempt in range(FIREWORKS_JSON_RETRY_ATTEMPTS):
        try:
            first_draft_dialogue = output_model.model_validate_json(response_json)
            break
        except ValidationError as e:
            if attempt == FIREWORKS_JSON_RETRY_ATTEMPTS - 1:  # Last attempt
                raise ValueError(
                    f"Failed to parse dialogue JSON after {FIREWORKS_JSON_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            error_message = (
                f"Failed to parse dialogue JSON (attempt {attempt + 1}): {e}"
            )
            # Re-call the LLM with the error message
            system_prompt_with_error = f"{system_prompt}\n\nPlease return a VALID JSON object. This was the earlier error: {error_message}"
            response = call_llm(system_prompt_with_error, input_text, output_model)
            response_json = response.choices[0].message.content
            first_draft_dialogue = output_model.model_validate_json(response_json)

    # Call the LLM a second time to improve the dialogue
    system_prompt_with_dialogue = f"{system_prompt}\n\nHere is the first draft of the dialogue you provided:\n\n{first_draft_dialogue}."

    # Validate the response
    for attempt in range(FIREWORKS_JSON_RETRY_ATTEMPTS):
        try:
            response = call_llm(
                system_prompt_with_dialogue,
                "Please improve the dialogue. Make it more natural and engaging.",
                output_model,
            )
            final_dialogue = output_model.model_validate_json(
                response.choices[0].message.content
            )
            break
        except ValidationError as e:
            if attempt == FIREWORKS_JSON_RETRY_ATTEMPTS - 1:  # Last attempt
                raise ValueError(
                    f"Failed to improve dialogue after {FIREWORKS_JSON_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            error_message = f"Failed to improve dialogue (attempt {attempt + 1}): {e}"
            system_prompt_with_dialogue += f"\n\nPlease return a VALID JSON object. This was the earlier error: {error_message}"
    return final_dialogue


def call_llm(system_prompt: str, text: str, dialogue_format: Any) -> Any:
    """Call the LLM with the given prompt and dialogue format."""
    response = fw_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=FIREWORKS_MODEL_ID,
        max_tokens=FIREWORKS_MAX_TOKENS,
        temperature=FIREWORKS_TEMPERATURE,
        response_format={
            "type": "json_object",
            "schema": dialogue_format.model_json_schema(),
        },
    )
    return response


def parse_url(url: str) -> str:
    """Parse the given URL and return the text content."""
    for attempt in range(JINA_RETRY_ATTEMPTS):
        try:
            full_url = f"{JINA_READER_URL}{url}"
            response = requests.get(full_url, timeout=60)
            response.raise_for_status()  # Raise an exception for bad status codes
            break
        except requests.RequestException as e:
            if attempt == JINA_RETRY_ATTEMPTS - 1:  # Last attempt
                raise ValueError(
                    f"Failed to fetch URL after {JINA_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            time.sleep(JINA_RETRY_DELAY)  # Wait for X second before retrying
    return response.text


def generate_podcast_audio(
    text: str, speaker: str, language: str, use_advanced_audio: bool, random_voice_number: int
) -> str:
    """Generate audio for podcast using TTS or advanced audio models."""
    if use_advanced_audio:
        return _use_suno_model(text, speaker, language, random_voice_number)
    else:
        return _use_melotts_api(text, speaker, language)


def _use_suno_model(text: str, speaker: str, language: str, random_voice_number: int) -> str:
    """Generate advanced audio using Bark."""
    host_voice_num = str(random_voice_number)
    guest_voice_num = str(random_voice_number + 1)
    audio_array = generate_audio(
        text,
        history_prompt=f"v2/{language}_speaker_{host_voice_num if speaker == 'Host (Jane)' else guest_voice_num}",
    )
    file_path = f"audio_{language}_{speaker}.mp3"
    write_wav(file_path, SAMPLE_RATE, audio_array)
    return file_path


def _use_melotts_api(text: str, speaker: str, language: str) -> str:
    """Generate audio using TTS model."""
    accent, speed = _get_melo_tts_params(speaker, language)

    for attempt in range(MELO_RETRY_ATTEMPTS):
        try:
            return hf_client.predict(
                text=text,
                language=language,
                speaker=accent,
                speed=speed,
                api_name=MELO_API_NAME,
            )
        except Exception as e:
            if attempt == MELO_RETRY_ATTEMPTS - 1:  # Last attempt
                raise  # Re-raise the last exception if all attempts fail
            time.sleep(MELO_RETRY_DELAY)  # Wait for X second before retrying


def _get_melo_tts_params(speaker: str, language: str) -> tuple[str, float]:
    """Get TTS parameters based on speaker and language."""
    if speaker == "Guest":
        accent = "EN-US" if language == "EN" else language
        speed = 0.9
    else:  # host
        accent = "EN-Default" if language == "EN" else language
        speed = (
            1.1 if language != "EN" else 1
        )  # if the language is not English, try speeding up so it'll sound different from the host
        # for non-English, there is only one voice
    return accent, speed