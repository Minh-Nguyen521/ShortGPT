from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from metrics import block_influence


class ShortEmbedding:

    def __init__(
        self,
        model_name_or_path: str,
        n_prune_layers: Optional[int] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model id or local path.
                                Works with any SentenceTransformer-compatible model.
            n_prune_layers: Number of layers to remove when calling remove_layers().
            device: Target device. Defaults to CUDA if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.st_model = SentenceTransformer(model_name_or_path, device=self.device)

        # The underlying HF model and tokenizer live inside the first module
        self._transformer = self.st_model[0]
        self._auto_model = self._transformer.auto_model
        self._tokenizer = self._transformer.tokenizer

        # Enable hidden states — set on config AND passed explicitly in forward()
        # to handle models that don't read config.output_hidden_states at runtime.
        self._auto_model.config.output_hidden_states = True

        self.n_prune_layers = n_prune_layers
        self.importances = [0.0 for _ in self._get_layers()]

    def _get_layers(self) -> nn.ModuleList:
        """Return the ModuleList of transformer layers from the underlying HF model."""
        m = self._auto_model
        # BERT-style
        if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
            return m.encoder.layer
        # Gemma / Llama-style
        if hasattr(m, "layers"):
            return m.layers
        # Wrapped one level deeper
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        raise AttributeError(
            "Cannot find encoder layers. Inspect model architecture and extend _get_layers()."
        )

    @torch.inference_mode()
    def eval_importance(
        self,
        sentences: List[str],
        batch_size: int = 32,
        angular: bool = False,
    ):
        """
        Accumulate layer-wise BI scores over a list of sentences.
        Uses the model's own tokenizer and mean pooling (matching SentenceTransformer behavior).

        Args:
            sentences: List of input strings.
            batch_size: Number of sentences per forward pass.
            angular: Whether to use angular distance instead of cosine-based BI.
        """
        self._auto_model.eval()

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._transformer.max_seq_length,
                return_tensors="pt",
            )
            # Only pass keys the model accepts; Gemma does not accept token_type_ids
            safe_keys = {"input_ids", "attention_mask"}
            encoded = {k: v.to(self.device) for k, v in encoded.items() if k in safe_keys}

            outputs = self._auto_model(**encoded, output_hidden_states=True)
            # hidden_states: tuple of (n_layers + 1) tensors shaped (B, S, D)
            # index 0 = token embeddings, index i+1 = output of transformer layer i
            hidden_states = outputs.hidden_states
            assert hidden_states is not None, (
                "Model did not return hidden_states. Ensure the underlying model supports output_hidden_states."
            )

            # Mean-pool each hidden state over non-padding tokens -> (B, 1, D)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            hiddens = [
                (h * attention_mask).sum(dim=1, keepdim=True) / attention_mask.sum(dim=1, keepdim=True)
                for h in hidden_states
            ]

            for i in range(len(self.importances)):
                self.importances[i] += block_influence(
                    hiddens[i],
                    hiddens[i + 1],
                    angular=angular,
                ).sum().cpu().item()

    def remove_layers(self, layers_to_remove: Optional[List[int]] = None) -> List[int]:
        """
        Remove the least important layers based on accumulated BI scores.

        Args:
            layers_to_remove: Explicit list of layer indices to remove.
                              If None, uses n_prune_layers least important layers.

        Returns:
            List of removed layer indices.
        """
        if layers_to_remove is None:
            assert self.n_prune_layers, "Set n_prune_layers or pass layers_to_remove explicitly."
            assert any(v > 0 for v in self.importances), "Run eval_importance() before remove_layers()."
            layers_to_remove = np.argsort(np.array(self.importances))[: self.n_prune_layers].tolist()

        layers = self._get_layers()
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del layers[layer_idx]
            except IndexError:
                print(f"Layer {layer_idx} does not exist, skipping.")
                return []

        self.importances = [0.0 for _ in self._get_layers()]
        return layers_to_remove

    def get_embeddings(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode sentences using the (possibly pruned) SentenceTransformer model.
        Pooling and normalization are handled by the SentenceTransformer pipeline.

        Args:
            sentences: List of input strings.
            batch_size: Number of sentences per forward pass.

        Returns:
            Numpy array of shape (N, D).
        """
        return self.st_model.encode(sentences, batch_size=batch_size, show_progress_bar=False)
