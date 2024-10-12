import asyncio
import websockets
import json
import base64
import os
import pyaudio

# OpenAI API configuration
API_KEY = os.environ.get("OPENAI_API_KEY")
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def main():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(URL, extra_headers=headers) as websocket:
        print("Connected to OpenAI Realtime API")

        # Set up PyAudio for input
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        # Send initial session configuration
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "You are a helpful AI assistant. Respond concisely.",
            }
        }))

        # Main interaction loop
        while True:
            # Record audio input
            print("Speak now...")
            frames = []
            for _ in range(0, int(RATE / CHUNK * 5)):  # Record for 5 seconds
                data = stream.read(CHUNK)
                frames.append(data)

            # Send audio to API
            audio_data = b''.join(frames)
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            await websocket.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "audio", "audio": audio_base64}]
                }
            }))

            # Process API response
            print("Processing response...")
            response_text = ""
            async for message in websocket:
                event = json.loads(message)
                if event['type'] == 'conversation.item.created':
                    if event['item']['type'] == 'message' and event['item']['role'] == 'assistant':
                        for content in event['item']['content']:
                            if content['type'] == 'text':
                                response_text += content['text']
                            elif content['type'] == 'audio':
                                # Here you would handle the audio response
                                # For simplicity, we're just acknowledging it
                                print("Received audio response")
                elif event['type'] == 'conversation.item.completed':
                    break

            print("Assistant:", response_text)

            # Ask if user wants to continue
            user_input = input("Continue? (y/n): ")
            if user_input.lower() != 'y':
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())