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

    def _capture_hidden_states(self, encoded: dict) -> List[torch.Tensor]:
        """
        Capture per-layer hidden states using forward hooks.
        Works even when output_hidden_states is not propagated by custom model code (e.g. Jina).
        Returns a list of length n_layers + 1: [embedding_output, layer_0_out, ..., layer_n_out].
        """
        layers = self._get_layers()
        captured = []
        hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                # Layer output can be a tuple (hidden, ...) or a plain tensor
                h = output[0] if isinstance(output, tuple) else output
                captured.append(h.detach())
            return hook

        # Capture the input to the first layer as the embedding output
        def input_hook(module, input, output):
            h = input[0] if isinstance(input, tuple) else input
            captured.insert(0, h.detach())

        hooks.append(layers[0].register_forward_pre_hook(
            lambda m, inp: captured.insert(0, (inp[0] if isinstance(inp, tuple) else inp).detach())
        ))
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        try:
            safe_keys = {"input_ids", "attention_mask"}
            safe_encoded = {k: v for k, v in encoded.items() if k in safe_keys}
            with torch.inference_mode():
                self._auto_model(**safe_encoded)
        finally:
            for h in hooks:
                h.remove()

        return captured

    @torch.inference_mode()
    def eval_importance(
        self,
        sentences: List[str],
        batch_size: int = 32,
        angular: bool = False,
    ):
        """
        Accumulate layer-wise BI scores over a list of sentences.
        Uses forward hooks to capture hidden states — works with custom model code (e.g. Jina).

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
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            hidden_states = self._capture_hidden_states(encoded)

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

        # Keep config in sync so saved model loads correctly
        if hasattr(self._auto_model.config, "num_hidden_layers"):
            self._auto_model.config.num_hidden_layers = len(self._get_layers())

        return layers_to_remove

    def _get_ffn(self, layer: nn.Module):
        """
        Return (gate_proj, up_proj, down_proj) for SwiGLU FFNs,
        or (fc1, None, fc2) for standard FFNs.
        Handles common naming conventions across model families.
        """
        mlp = getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
        if mlp is None:
            raise AttributeError(f"Cannot find FFN in layer: {type(layer)}")

        # SwiGLU style (Jina, Llama, Gemma)
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            return mlp.gate_proj, mlp.up_proj, mlp.down_proj

        # Standard style (BERT, RoBERTa)
        if hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            return mlp.fc1, None, mlp.fc2

        # Some models use dense / dense_h_to_4h
        if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "dense_4h_to_h"):
            return mlp.dense_h_to_4h, None, mlp.dense_4h_to_h

        raise AttributeError(f"Unknown FFN structure in: {type(mlp)}")

    def _get_ffn_mlp(self, layer: nn.Module) -> nn.Module:
        """Return the full MLP/FFN module from a transformer layer."""
        mlp = getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
        if mlp is None:
            raise AttributeError(f"Cannot find FFN in layer: {type(layer)}")
        return mlp

    def eval_ffn_importance(
        self,
        sentences: List[str],
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """
        Measure per-neuron importance in each layer's FFN.

        For SwiGLU: hooks the input to down_proj which is silu(gate) * up — the actual
        values flowing into the contracting projection, not just pre-activation gate values.
        For standard FFN: hooks the output of fc1 (post-activation).

        Returns:
            List of numpy arrays, one per layer, each of shape (intermediate_size,).
            Higher value = more important neuron.
        """
        self._auto_model.eval()
        layers = self._get_layers()
        n_layers = len(layers)
        neuron_scores: List[Optional[np.ndarray]] = [None] * n_layers
        hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Hook on down_proj: input[0] is (B, S, intermediate_size)
                acts = input[0] if isinstance(input, tuple) else input
                score = acts.detach().abs().mean(dim=(0, 1)).cpu().float().numpy()
                if neuron_scores[layer_idx] is None:
                    neuron_scores[layer_idx] = score
                else:
                    neuron_scores[layer_idx] += score
            return hook

        for i, layer in enumerate(layers):
            _, _, down = self._get_ffn(layer)
            # Hook on down_proj input = actual neuron activations feeding into contraction
            hooks.append(down.register_forward_hook(make_hook(i)))

        try:
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self._transformer.max_seq_length,
                    return_tensors="pt",
                )
                safe_keys = {"input_ids", "attention_mask"}
                encoded = {k: v.to(self.device) for k, v in encoded.items() if k in safe_keys}
                self._auto_model(**encoded)
        finally:
            for h in hooks:
                h.remove()

        return [s if s is not None else np.array([]) for s in neuron_scores]

    def prune_ffn(self, neuron_scores: List[np.ndarray], prune_ratio: float = 0.2):
        """
        Remove the least active FFN neurons from every layer based on neuron_scores.

        For SwiGLU: slices gate_proj, up_proj (rows) and down_proj (columns).
        For standard FFN: slices fc1 (rows) and fc2 (columns).

        Args:
            neuron_scores: Output of eval_ffn_importance().
            prune_ratio: Fraction of neurons to remove per layer (default 0.2 = 20%).
        """
        layers = self._get_layers()
        total_before = total_after = 0

        for i, layer in enumerate(layers):
            scores = neuron_scores[i]
            if scores is None or len(scores) == 0:
                continue

            n_keep = max(1, int(len(scores) * (1 - prune_ratio)))
            keep_idx = np.argsort(scores)[-n_keep:]  # keep highest scoring
            keep_idx = np.sort(keep_idx)
            keep_t = torch.tensor(keep_idx, device=self.device)

            gate, up, down = self._get_ffn(layer)
            total_before += gate.weight.shape[0]

            # Slice expanding projections (rows = neurons)
            with torch.no_grad():
                gate.weight = nn.Parameter(gate.weight[keep_t])
                if gate.bias is not None:
                    gate.bias = nn.Parameter(gate.bias[keep_t])

                if up is not None:
                    up.weight = nn.Parameter(up.weight[keep_t])
                    if up.bias is not None:
                        up.bias = nn.Parameter(up.bias[keep_t])

                # Slice contracting projection (columns = neurons)
                down.weight = nn.Parameter(down.weight[:, keep_t])

            # Update linear layer dimensions
            gate.out_features = n_keep
            if up is not None:
                up.out_features = n_keep
            down.in_features = n_keep

            total_after += n_keep

        # Update config
        if hasattr(self._auto_model.config, "intermediate_size"):
            # Set to the new size of the first layer as representative
            gate, _, _ = self._get_ffn(layers[0])
            self._auto_model.config.intermediate_size = gate.out_features

        print(f"FFN neurons: {total_before} → {total_after} ({total_before - total_after} removed)")

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
