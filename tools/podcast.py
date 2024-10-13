from pathlib import Path
from openai import OpenAI
from pydub import AudioSegment
import os
from typing import Optional

client = OpenAI()

def write_podcast_script(txt_file: Optional[str] = None, prompt: Optional[str] = None ) -> str:
    assert txt_file is not None or prompt is not None, "txt_file and prompt cannot be both None"
    
    if txt_file is not None:
        with open(txt_file, "r") as file:
            prompt = file.read()
            
    # Generate podcast script using OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a podcast host discussing a code repository. Create a script for a short podcast with two hosts: Host A and Host B. Host A is Steve Jobs and Host B is Elon Musk. The script should be informative and engaging."},
            {"role": "user", "content": f"Create a podcast script based on this repository description:\n\n{prompt}"}
        ]
    )

    raw_podcast_script = response.choices[0].message.content
    
    return raw_podcast_script

match_a_pattern = ["**Host A (Steve Jobs):**", "**Steve:**", "**Steve Jobs (Host A):**", "**Steve Jobs (Host A):**", "Host A: Steve Jobs:", "**Host A: Steve Jobs:**", "**Steve Jobs (Host A)**:", "**Steve Jobs:**", "**Steve Jobs**:", "**Steve Jobs**", "Steve Jobs:", "Host A:", "**Host A:**", "**Host A**:"]
match_b_pattern = ["**Host B (Elon Musk):**", "**Elon:**", "**Elon Musk (Host B):**", "**Elon Musk (Host B):**", "Host B: Elon Musk:", "**Host B: Elon Musk:**", "**Elon Musk (Host B)**:", "**Elon Musk:**", "**Elon Musk**:", "**Elon Musk**", "Elon Musk:", "Host B:", "**Host B:**", "**Host B**:"]

def get_line_content(line, pattern):
    content = line.split(pattern)[-1].split(":")[-1].strip()
    content = content.replace("Steve Jobs", "Steve")
    content = content.replace("Elon Musk", "Elon")
    return content


def parse_script(raw_script):
    parsed_scripts = []

    for line in raw_script.split("\n"):
        current_speaker = None
        
        # sort pattern by length
        a_patterns = sorted(match_a_pattern, key=len, reverse=True)
        b_patterns = sorted(match_b_pattern, key=len, reverse=True)
        for pattern in a_patterns:
            if pattern in line:
                content = get_line_content(line, pattern)
                current_speaker = "Host A"
                break
            
        for pattern in b_patterns:
            if pattern in line:
                content = get_line_content(line, pattern)
                current_speaker = "Host B"
                break
                
        if current_speaker:
            parsed_scripts.append({"speaker": current_speaker, "text": content})
    
    return parsed_scripts

def generate_podcast_audio(parsed_podcast: dict, name: str, output_dir: str):
    audio_files = []

    for i, part in enumerate(parsed_podcast):
        voice = "alloy" if part["speaker"] == "Host A" else "echo"
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=part["text"]
        )
        
        file_name = f"part_{i}.mp3"
        with open(file_name, "wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
        audio_files.append(file_name)

    # Combine audio files
    combined = AudioSegment.empty()
    for file in audio_files:
        segment = AudioSegment.from_mp3(file)
        combined += segment

    output_file = os.path.join(output_dir, f"podcast_{name}.mp3")
    combined.export(output_file, format="mp3")

    print(f"Podcast created and saved as {output_file}")

    # Clean up temporary files
    for file in audio_files:
        os.remove(file)
        
    return output_file