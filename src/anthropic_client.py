import os
import time

# IMPORTANT: Must be Anthropic >= 0.4
# pip install --upgrade anthropic
from anthropic import Anthropic, APIError, RateLimitError

class AnthropicClient:
    def __init__(self, api_key=None, max_retries=3, retry_delay=1):
        # Get API key from argument or environment
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # Create the new Anthropic client (>=0.4)
        self.client = Anthropic(api_key=self.api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def call_model(self, model: str, messages: list, system: str, max_tokens=8000, temperature=0):
        """
        Generic wrapper around the new Anthropic 'messages.create' call.
        For Anthropic >= 0.4.x, you do:

            client.messages.create(
                model=model,
                system=system,
                messages=messages,
                max_tokens=...,
                temperature=...
            )

        Returns an object whose text is in 'response.completion'.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.messages.create(
                    model=model,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response
            except RateLimitError as e:
                retries += 1
                if retries == self.max_retries:
                    raise Exception(f"Rate limit exceeded after {retries} retries") from e
                print(f"Rate limit hit, retrying in {self.retry_delay} seconds...")
                # Exponential backoff
                time.sleep(self.retry_delay * (2 ** (retries - 1)))
            except APIError as e:
                retries += 1
                if retries == self.max_retries:
                    raise Exception(f"API error after {retries} retries: {str(e)}") from e
                print(f"API error occurred, retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            except Exception as e:
                # Any other unexpected error
                raise Exception(f"Unexpected error during API call: {str(e)}") from e

    def is_new_document(self, prev_image_b64: str, curr_image_b64: str) -> bool:
        """
        Determine if 'curr_image_b64' starts a new document compared to 'prev_image_b64'.
        This code passes base64 images as TEXT. Claude won't parse them as real images,
        but at least it won't error out. 
        """
        try:
            # Anthropic uses a 'system' parameter for the system prompt
            system_prompt = (
                "You are a document segmentation assistant. "
                "Given two consecutive pages (previous and current) in base64 images, "
                "decide if the current page starts a new document. "
                "Output only 'YES' or 'NO'."
            )

            # The new messages format: a list of { "role": "user"/"system"/"assistant", "content": "..." }
            # Here, we'll embed the base64 images into the user message as text
            user_text = (
                f"PREVIOUS PAGE (base64):\n{prev_image_b64}\n\n"
                f"CURRENT PAGE (base64):\n{curr_image_b64}\n\n"
                "Does the current page start a new document? Respond YES or NO."
            )

            messages = [
                {
                    "role": "user",
                    "content": user_text
                }
            ]

            resp = self.call_model(
                model="claude-3-5-sonnet-latest",  # or whichever model you prefer
                messages=messages,
                system=system_prompt,
                max_tokens=8000,
                temperature=0
            )

            # The new library returns an object with a `.completion` attribute
            answer = resp.completion.strip().upper()
            return "YES" in answer
        except (RateLimitError, APIError):
            # Let these propagate up since 'call_model' handles retries
            raise
        except Exception as e:
            # Handle any other unexpected errors (like malformed responses, etc.)
            print(f"Unexpected error in is_new_document: {str(e)}")
            # Default to NO on error, or you could `return False`
            return False
