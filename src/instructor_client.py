import openai
from anthropic import Anthropic

class DocumentMetadata:
    """
    A simple data structure to hold document metadata.
    Adjust fields as needed for your use case.
    """
    def __init__(self, title, date, summary, tags=None):
        self.title = title
        self.date = date
        self.summary = summary
        self.tags = tags if tags is not None else []

class InstructorClient:
    """
    A client that can wrap either OpenAI or Anthropic calls
    to orchestrate summarizing, instructing, or extracting metadata.
    """
    def __init__(self, model_client, model_provider="openai"):
        """
        - model_client: 
            If model_provider='anthropic', this might be an `Anthropic` instance 
            or a wrapper that calls anthropic.
            If model_provider='openai', you won't strictly need `model_client`, 
            since openai is used globally, but you can still pass in a reference.
        - model_provider: "openai" or "anthropic"
        """
        self.model_client = model_client
        self.model_provider = model_provider.lower()

    def get_summary_openai(self, text: str) -> str:
        """
        Summarize using OpenAI completions (text-davinci-003 or similar).
        """
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Summarize the following text:\n\n{text}\n\nSummary:",
                max_tokens=150,
                temperature=0.7
            )
            summary = response["choices"][0]["text"].strip()
            return summary
        except Exception as e:
            print(f"[OpenAI] Error while generating summary: {str(e)}")
            return ""

    def get_summary_anthropic(self, text: str) -> str:
        """
        Summarize using Anthropic's chat API (>=0.4.x).
        You must have an Anthropic instance that supports .chat(...) or .messages.create(...).
        """
        try:
            # We'll do a simple user prompt with the text, 
            # and ask the assistant for a summary.
            if not isinstance(self.model_client, Anthropic):
                raise ValueError("model_client must be an Anthropic instance for Anthropics mode.")

            response = self.model_client.chat(
                model="claude-2",  # or "claude-instant-v1" etc.
                system="You are a helpful assistant that summarizes documents.",
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize the following text:\n\n{text}\n\nSummary:"
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            # In the new Anthropic library, the summary is at response.completion
            summary = response.completion.strip()
            return summary
        except Exception as e:
            print(f"[Anthropic] Error while generating summary: {str(e)}")
            return ""

    def get_summary(self, text: str) -> str:
        """
        Route to the correct summarization method depending on the provider.
        """
        if self.model_provider == "openai":
            return self.get_summary_openai(text)
        elif self.model_provider == "anthropic":
            return self.get_summary_anthropic(text)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

    def analyze_document(self, text: str) -> DocumentMetadata:
        """
        Analyze a document (generate summary, possibly more)
        and return DocumentMetadata.
        """
        summary = self.get_summary(text)
        # Placeholder logic for title, date, and tags
        return DocumentMetadata(
            title="Untitled Document",
            date="Unknown Date",
            summary=summary,
            tags=["example", self.model_provider]
        )

    def extract_metadata(self, text: str) -> DocumentMetadata:
        """
        Wrapper for `analyze_document` to align with expected method usage.
        """
        return self.analyze_document(text)
