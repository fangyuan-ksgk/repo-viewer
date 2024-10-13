import anthropic
from typing import Union, List
from openai import OpenAI

anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()

def get_text_content(query: Union[str, List[str]]):
    if isinstance(query, list):
        return [{"type": "text", "text": q} for q in query]
    else:
        return [{"type": "text", "text": query}]

def get_claude_response(query: Union[str, List[str]], img: str = None, img_type: str = None, system_prompt: str = "You are a helpful assistant."):
    """ 
    Claude response with query and image input
    """
    text_content = get_text_content(query)
    
    if img is not None:
        img_content = [{"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img}}]
        content = img_content + text_content
    else:
        content = text_content
    
    message = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        system=system_prompt,
    )
    return message.content[0].text


def get_openai_response(query: Union[str, List[str]], img: str = None, img_type: str = "image/jpeg", system_prompt: str = "You are a helpful assistant."):
    """
    OpenAI response with query and optional image input
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    content = []
    if isinstance(query, str):
        content.append({"type": "text", "text": query})
    elif isinstance(query, list):
        for q in query:
            if isinstance(q, dict) and "role" in q and "content" in q:
                content.append({"type": "text", "text": q["content"]})
            else:
                content.append({"type": "text", "text": q})
    
    if img is not None:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img_type};base64,{img}"
            }
        })
    
    messages.append({"role": "user", "content": content})
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content