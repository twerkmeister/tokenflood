from typing import List, Dict

MessageList = List[Dict[str, str]]

chat_schema = {
    "type": "array",
    "description": "A list of chat messages",
    "items": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["system", "user", "assistant", "tool"]},
            "content": {"type": "string"},
        },
        "required": ["role", "content"],
        "additionalProperties": False,
    },
}
