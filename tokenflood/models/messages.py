from typing import Dict, List

MessageList = List[Dict[str, str]]


def create_message_list_from_prompt(prompt: str) -> MessageList:
    return [{"role": "user", "content": prompt}]
