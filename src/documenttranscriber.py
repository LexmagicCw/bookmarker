import os  # Add this at the top of the file
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

class DocumentTranscriber:
    def __init__(self, model_client, provider="openai"):
        self.provider = provider
        if provider == "anthropic":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "openai":
            self.client = model_client
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def transcribe_page(self, image_b64: str, extracted_text: str) -> str:
        """Transcribe a single page to markdown format."""
        try:
            if self.provider == "anthropic":
                prompt = f"""{HUMAN_PROMPT}Please transcribe this document page exactly as it appears, preserving all formatting.
                Format your response as markdown, including:
                - Proper headings (# for main titles, ## for subtitles, etc.)
                - Bold and italic text where appropriate
                - Lists (numbered and bulleted) as they appear
                - Indentation and spacing
                - Tables if present
                
                OCR-extracted text for reference:\n{extracted_text}\n{AI_PROMPT}"""

                response = self.client.completions.create(
                    model="claude-2",
                    prompt=prompt,
                    max_tokens_to_sample=8000,
                )
                return response["completion"].strip()

            elif self.provider == "openai":
                messages = [
                    {
                        "role": "user",
                        "content": f"""Please transcribe this document page exactly as it appears, preserving all formatting.
                        Format your response as markdown, including:
                        - Proper headings (# for main titles, ## for subtitles, etc.)
                        - Bold and italic text where appropriate
                        - Lists (numbered and bulleted) as they appear
                        - Indentation and spacing
                        - Tables if present

                        OCR-extracted text for reference:\n{extracted_text}"""
                    }
                ]
                response = self.client.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=8000,
                )
                return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error transcribing page: {str(e)}")
            return ""
