import torch
from transformers import CLIPTokenizer, CLIPTextModel

class TextToEmbedding:
    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cuda'):
        self.device = device
        # Load the tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts):
        """Converts a list of texts to latent embeddings."""
        # Tokenize text inputs
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        # Pass through the text encoder
        outputs = self.text_encoder(**inputs)
        # Return the text embeddings
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling across token embeddings
    
if __name__ == '__main__':
    tokenizer = TextToEmbedding()
    result = tokenizer.encode('The weather is lovely today.')
    print(result.shape)
    print(result)