import torch
import torch.nn as nn


class CausalMemoryIntervention(nn.Module):
    """
    Módulo de intervención causal sobre memorias de eventos.

    Objetivo:
    - intervenir memorias temporales ya construidas
    - estimar qué tanto influye cada memoria en la predicción final
    - producir un score causal por evento

    Intervenciones soportadas:
    - zero: reemplaza la memoria intervenida por ceros
    - mean: reemplaza la memoria intervenida por la media del resto
    """

    def __init__(self, intervention_type="zero"):
        super().__init__()
        self.intervention_type = intervention_type

    def intervene(self, memory_bank, index):
        """
        memory_bank: Tensor [B, T, D]
        index: índice temporal a intervenir

        retorna:
            intervened_memory_bank: Tensor [B, T, D]
        """
        intervened = memory_bank.clone()

        if self.intervention_type == "zero":
            intervened[:, index, :] = 0.0

        elif self.intervention_type == "mean":
            B, T, D = memory_bank.shape
            if T == 1:
                intervened[:, index, :] = 0.0
            else:
                mask = torch.ones(T, dtype=torch.bool, device=memory_bank.device)
                mask[index] = False
                mean_memory = memory_bank[:, mask, :].mean(dim=1)  # [B, D]
                intervened[:, index, :] = mean_memory

        else:
            raise ValueError(f"Tipo de intervención no soportado: {self.intervention_type}")

        return intervened

    def compute_effect(self, original_probs, intervened_probs, target_idx=None):
        """
        original_probs:   [B, C]
        intervened_probs: [B, C]
        target_idx:
            - None: usa la clase predicha originalmente
            - Tensor [B]: usa esos índices de clase

        retorna:
            causal_effect: [B]
        """
        if target_idx is None:
            target_idx = torch.argmax(original_probs, dim=1)

        B = original_probs.size(0)
        row_idx = torch.arange(B, device=original_probs.device)

        original_selected = original_probs[row_idx, target_idx]
        intervened_selected = intervened_probs[row_idx, target_idx]

        causal_effect = original_selected - intervened_selected
        return causal_effect

    def rank_event_importance(self, original_probs, intervened_prob_list, target_idx=None):
        """
        original_probs: [B, C]
        intervened_prob_list: lista de T tensores [B, C]

        retorna:
            effects: [B, T]
        """
        effects = []
        for probs_i in intervened_prob_list:
            eff_i = self.compute_effect(original_probs, probs_i, target_idx=target_idx)
            effects.append(eff_i)

        effects = torch.stack(effects, dim=1)  # [B, T]
        return effects

    def forward(self, memory_bank, qa_inference_fn, target_idx=None):
        """
        memory_bank: [B, T, D]

        qa_inference_fn:
            función que recibe un memory_bank [B, T, D]
            y retorna logits [B, C]

        target_idx:
            opcional, clases objetivo [B]

        retorna:
            dict con:
            - original_logits
            - original_probs
            - causal_effects [B, T]
        """
        original_logits = qa_inference_fn(memory_bank)
        original_probs = torch.softmax(original_logits, dim=-1)

        B, T, D = memory_bank.shape
        intervened_prob_list = []

        for t in range(T):
            intervened_memory = self.intervene(memory_bank, t)
            intervened_logits = qa_inference_fn(intervened_memory)
            intervened_probs = torch.softmax(intervened_logits, dim=-1)
            intervened_prob_list.append(intervened_probs)

        causal_effects = self.rank_event_importance(
            original_probs,
            intervened_prob_list,
            target_idx=target_idx
        )

        return {
            "original_logits": original_logits,
            "original_probs": original_probs,
            "causal_effects": causal_effects
        }