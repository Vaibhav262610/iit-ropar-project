from pathway.xpacks.llm.embedders import BaseEmbedder
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np


class SpladeEmbedder(BaseEmbedder):
    def __init__(self, model: str, **kwargs):
        super().__init__()
        with torch.no_grad():
            self.model = AutoModelForMaskedLM.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def __wrapped__(self, input: str, **kwargs) -> np.ndarray:
        with torch.no_grad():
            tokens = self.tokenizer(input, return_tensors="pt")
            output = self.model(**tokens)
            vec = torch.max(
                torch.log(1 + torch.relu(output.logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0].squeeze()

            return vec.cpu().numpy()
