import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMemoryRefiner(nn.Module):
    """
    Refinamiento causal de memorias.

    Objetivo:
    1. Predecir cuánto influye cada memoria sobre la respuesta.
    2. Aprender esos scores usando como teacher los efectos obtenidos
       mediante intervenciones reales.
    3. Reponderar las memorias sin tener que ejecutar 10 intervenciones
       durante la inferencia final.

    Inputs:
        memory_bank:
            [B, T, D]

        qa_logits:
            [B, C]

            Son los logits de la predicción original del modelo.

    Outputs:
        refined_memory:
            [B, T, D]

        predicted_effects:
            [B, T]

        memory_weights:
            [B, T]
    """

    def __init__(
        self,
        hidden_dim=512,
        num_answers=4,
        predictor_dim=256,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_answers = num_answers

        # Predictor de influencia causal
        #
        # Cada memoria se combina con el contexto de respuesta
        # del modelo.

        self.effect_predictor = nn.Sequential(
            nn.Linear(
                hidden_dim + num_answers,
                predictor_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                predictor_dim,
                predictor_dim // 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                predictor_dim // 2,
                1
            )
        )

        # Gate residual aprendible
        #
        # Inicia en cero para que inicialmente:
        #
        # refined_memory == memory_bank
        #
        # exactamente igual que hicimos con TCMR temporal.
        self.beta = nn.Parameter(
            torch.tensor(0.0)
        )

    # PREDICCIÓN DEL EFECTO CAUSAL


    def predict_effects(
        self,
        memory_bank,
        qa_logits
    ):
        """
        Predice un efecto para cada memoria.

        memory_bank:
            [B, T, D]

        qa_logits:
            [B, C]

        return:
            [B, T]
        """

        B, T, D = memory_bank.shape

        # Convertir logits de QA a probabilidades.
        #
        # Esto introduce contexto relacionado con la pregunta
        # y la respuesta actual del modelo.
        qa_context = F.softmax(
            qa_logits,
            dim=-1
        )

        # [B, C]
        # ->
        # [B, T, C]
        qa_context = (
            qa_context
            .unsqueeze(1)
            .expand(
                -1,
                T,
                -1
            )
        )

        # Combinar:
        #
        # memoria del evento
        # +
        # contexto de respuesta
        predictor_input = torch.cat(
            [
                memory_bank,
                qa_context
            ],
            dim=-1
        )

        # [B, T, D+C]
        # ->
        # [B, T, 1]
        predicted_effects = (
            self.effect_predictor(
                predictor_input
            )
        )

        # [B, T, 1]
        # ->
        # [B, T]
        predicted_effects = (
            predicted_effects
            .squeeze(-1)
        )

        return predicted_effects

    # CONVERTIR EFECTOS EN PESOS

    def effects_to_weights(
        self,
        predicted_effects
    ):
        """
        Convierte los efectos predichos en pesos para cada memoria.

        El centrado hace que la comparación sea relativa
        entre las memorias de un mismo video/pregunta.
        """

        centered_effects = (
            predicted_effects
            -
            predicted_effects.mean(
                dim=1,
                keepdim=True
            )
        )

        # beta = 0
        # ->
        # todos los pesos son exactamente 1
        memory_weights = (
            1.0
            +
            torch.tanh(
                self.beta
            )
            *
            torch.tanh(
                centered_effects
            )
        )

        return memory_weights


    # FORWARD


    def forward(
        self,
        memory_bank,
        qa_logits,
        return_aux=False
    ):
        """
        Ejecuta predicción de influencia + refinamiento.
        """


        # 1. Predecir efecto causal


        predicted_effects = (
            self.predict_effects(
                memory_bank,
                qa_logits
            )
        )


        # 2. Convertir efectos en pesos


        memory_weights = (
            self.effects_to_weights(
                predicted_effects
            )
        )


        # 3. Reponderar memorias


        refined_memory = (
            memory_bank
            *
            memory_weights.unsqueeze(-1)
        )


        # Información auxiliar para entrenamiento/análisis


        if return_aux:

            return (
                refined_memory,
                predicted_effects,
                memory_weights
            )

        return refined_memory