import os
import requests
import json
import base64
import io
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

class OpenAILLMNode:
    def _determine_max_tokens_param(self, model, endpoint):
        """Return the appropriate parameter name for controlling completion length."""
        if endpoint and endpoint.rstrip("/").endswith("/responses"):
            return "max_output_tokens"

        modern_prefixes = (
            "gpt-4.1",
            "gpt-4o",
            "o4",
            "o3",
            "gpt-4-turbo",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "gpt-4o-realtime",
            "gpt-4.1-realtime",
        )

        legacy_prefixes = (
            "gpt-3.5",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-1106",
            "gpt-4-turbo-preview",
        )

        if any(model.startswith(prefix) for prefix in legacy_prefixes):
            return "max_tokens"

        if any(model.startswith(prefix) for prefix in modern_prefixes):
            return "max_completion_tokens"

        # Default to the modern parameter to align with current API guidance
        return "max_completion_tokens"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "Enter your prompt here..."
                }),
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": "https://api.openai.com/v1/chat/completions",
                    "placeholder": "OpenAI-compatible endpoint URL"
                }),
                "api_token": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Your API token"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "gpt-4-vision-preview",
                    "placeholder": "Model name"
                }),
                "max_tokens": ("INT", {
                    "default": 150,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "image_detail": (["low", "high", "auto"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "LLM"
    
    def _encode_image_to_base64(self, image_tensor):
        """Convert ComfyUI image tensor to base64 encoded string"""
        try:
            # ComfyUI images are typically in format [batch, height, width, channels]
            if len(image_tensor.shape) == 4:
                # Take first image from batch
                image_array = image_tensor[0]
            else:
                image_array = image_tensor

            # Ensure we are working with a numpy array
            if torch is not None and isinstance(image_array, torch.Tensor):
                image_array = image_array.detach().cpu().numpy()
            elif not isinstance(image_array, np.ndarray):
                image_array = np.asarray(image_array)

            # Convert from 0-1 float to 0-255 uint8 when needed
            if image_array.max() <= 1.0:
                image_array = image_array * 255.0

            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            # Convert numpy array to PIL Image
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image_array, 'RGB')
            elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA image
                pil_image = Image.fromarray(image_array, 'RGBA')
            else:
                # Grayscale or other format
                pil_image = Image.fromarray(image_array)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            raise Exception(f"Failed to encode image: {str(e)}")
    
    def _resolve_api_token(self, api_token):
        token = (api_token or "").strip()
        if token:
            return token

        env_token = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("OPENAI_TOKEN")
        )
        return (env_token or "").strip()

    def generate_text(self, prompt, endpoint, api_token, model="gpt-4-vision-preview", max_tokens=150, temperature=0.7, image=None, image_detail="auto"):
        try:
            resolved_token = self._resolve_api_token(api_token)
            if not resolved_token:
                return ("Error: Missing API token. Provide one in the node or set OPENAI_API_KEY / OPENAI_COMPAT_API_KEY.",)

            headers = {
                "Authorization": f"Bearer {resolved_token}",
                "Content-Type": "application/json"
            }
            
            # Construct message content
            if image is not None:
                # Encode image to base64
                image_data_url = self._encode_image_to_base64(image)
                
                # Create multimodal message with both text and image
                message_content = [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": image_detail
                        }
                    }
                ]
            else:
                # Text-only message
                message_content = prompt
            
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message_content}
                ],
                "temperature": temperature
            }

            tokens_param = self._determine_max_tokens_param(model, endpoint)
            data[tokens_param] = max_tokens
            
            response = requests.post(endpoint, headers=headers, json=data, timeout=30)

            if response.status_code >= 400:
                error_msg = f"{response.status_code} {response.reason}"
                try:
                    error_json = response.json()
                    if isinstance(error_json, dict):
                        if "error" in error_json:
                            api_error = error_json["error"]
                            if isinstance(api_error, dict):
                                detail = api_error.get("message") or api_error.get("error") or api_error
                                error_msg = f"{error_msg}: {detail}"
                            else:
                                error_msg = f"{error_msg}: {api_error}"
                        else:
                            error_msg = f"{error_msg}: {error_json}"
                    else:
                        error_msg = f"{error_msg}: {error_json}"
                except json.JSONDecodeError:
                    error_msg = f"{error_msg}: {response.text}"
                return (f"Request Error: {error_msg}",)

            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return (content,)
            else:
                return ("Error: No response content found",)
                
        except requests.exceptions.RequestException as e:
            return (f"Request Error: {str(e)}",)
        except json.JSONDecodeError as e:
            return (f"JSON Error: {str(e)}",)
        except Exception as e:
            return (f"Error: {str(e)}",)