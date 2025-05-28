from .openai_llm_node import OpenAILLMNode

NODE_CLASS_MAPPINGS = {
    "OpenAILLMNode": OpenAILLMNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAILLMNode": "OpenAI Compatible LLM"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']