import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalRelationEncoder(nn.Module):
    """
    Refinamiento temporal explícito de memorias de eventos.

    Recibe:
        memory_prompt:
            [batch_size, num_queries, hidden_dim]

        pred_spans:
            [batch_size, num_queries, 2]

            Cada span está representado como:
                (center, width)

            y está normalizado entre 0 y 1.

    Relaciones utilizadas:
        0 -> SELF
        1 -> BEFORE
        2 -> AFTER
        3 -> OVERLAP
    """

    SELF = 0
    BEFORE = 1
    AFTER = 2
    OVERLAP = 3

    def __init__(
        self,
        hidden_dim=512,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Embedding aprendible para cada relación temporal

        self.relation_embedding = nn.Embedding(
            4,
            hidden_dim
        )

        # Proyecciones para atención relacional

        self.query_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.key_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.value_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        # Proyecciones específicas para las relaciones


        self.relation_key_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.relation_value_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        # Salida de la atención temporal


        self.output_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )


        # Feed Forward Network


        self.ffn = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim * 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim * 2,
                hidden_dim
            )
        )


        # Normalización y dropout


        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Gate aprendible.
        # Se inicializa en 0 para que, antes del entrenamiento,
        # el módulo reproduzca exactamente las memorias originales.
        self.alpha = nn.Parameter(
            torch.tensor(0.0)
        )

    # CONVERSIÓN DE SPANS


    def spans_to_start_end(
        self,
        pred_spans
    ):
        """
        Convierte:

            (center, width)

        a:

            (start, end)

        Ejemplo:

            center = 0.5
            width  = 0.2

            start = 0.4
            end   = 0.6
        """

        center = pred_spans[..., 0]
        width = pred_spans[..., 1]

        start = center - width / 2.0
        end = center + width / 2.0

        # Mantener límites normalizados
        start = torch.clamp(
            start,
            min=0.0,
            max=1.0
        )

        end = torch.clamp(
            end,
            min=0.0,
            max=1.0
        )

        return start, end


    # MATRIZ DE RELACIONES TEMPORALES


    def build_relation_matrix(
        self,
        pred_spans
    ):
        """
        Construye una matriz:

            [B, Q, Q]

        donde cada posición [i, j] describe la relación
        temporal del evento i respecto al evento j.

        Ejemplo:

                    E1      E2      E3

            E1     SELF   BEFORE  OVERLAP
            E2     AFTER   SELF   AFTER
            E3   OVERLAP  BEFORE   SELF
        """

        start, end = self.spans_to_start_end(
            pred_spans
        )

        batch_size = pred_spans.size(0)
        num_queries = pred_spans.size(1)


        # Comparación evento i contra evento j


        start_i = start.unsqueeze(2)
        end_i = end.unsqueeze(2)

        start_j = start.unsqueeze(1)
        end_j = end.unsqueeze(1)

        # Inicialmente asumimos OVERLAP
        relations = torch.full(
            (
                batch_size,
                num_queries,
                num_queries
            ),
            self.OVERLAP,
            dtype=torch.long,
            device=pred_spans.device
        )


        # BEFORE
        #
        # El evento i termina antes de que comience j


        before_mask = (
            end_i < start_j
        )

        relations = torch.where(
            before_mask,
            torch.full_like(
                relations,
                self.BEFORE
            ),
            relations
        )

        # AFTER
        #
        # El evento i comienza después de que termina j


        after_mask = (
            start_i > end_j
        )

        relations = torch.where(
            after_mask,
            torch.full_like(
                relations,
                self.AFTER
            ),
            relations
        )


        # SELF


        identity = torch.eye(
            num_queries,
            dtype=torch.bool,
            device=pred_spans.device
        )

        identity = identity.unsqueeze(0)

        relations = torch.where(
            identity,
            torch.full_like(
                relations,
                self.SELF
            ),
            relations
        )

        return relations


    # FORWARD


    def forward(
        self,
        memory_prompt,
        pred_spans,
        return_aux=False
    ):
        """
        Refina las memorias utilizando relaciones temporales
        explícitas entre los eventos.
        """


        # 1. Crear matriz temporal
   

        relation_ids = self.build_relation_matrix(
            pred_spans
        )

        # [B, Q, Q, D]
        relation_embeddings = (
            self.relation_embedding(
                relation_ids
            )
        )


        # 2. Proyectar memorias


        query = self.query_proj(
            memory_prompt
        )

        key = self.key_proj(
            memory_prompt
        )

        value = self.value_proj(
            memory_prompt
        )


        # 3. Preparar dimensiones para comparación


        # Evento que recibe información
        query_i = query.unsqueeze(2)

        # Evento del que viene información
        key_j = key.unsqueeze(1)

        # 4. Incorporar relación temporal al key


        relation_keys = (
            self.relation_key_proj(
                relation_embeddings
            )
        )

        relational_keys = (
            key_j +
            relation_keys
        )


        # 5. Attention score


        attention_scores = (
            query_i *
            relational_keys
        ).sum(dim=-1)

        attention_scores = (
            attention_scores /
            math.sqrt(
                self.hidden_dim
            )
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )


        # 6. Incorporar relación temporal a los values


        value_j = value.unsqueeze(1)

        relation_values = (
            self.relation_value_proj(
                relation_embeddings
            )
        )

        relational_values = (
            value_j +
            relation_values
        )


        # 7. Agregación de eventos relacionados


        temporal_context = torch.sum(
            attention_weights.unsqueeze(-1)
            * relational_values,
            dim=2
        )

        temporal_context = self.output_proj(
            temporal_context
        )


        # 8. Residual connection


        refined_memory = self.norm1(
            memory_prompt
            +
            self.dropout(
                temporal_context
            )
        )


        # 9. Feed Forward


        ffn_output = self.ffn(
            refined_memory
        )

        refined_memory = self.norm2(
            refined_memory
            +
            self.dropout(
                ffn_output
            )
        )

        # ----------------------------------------------------
        # 10. Residual gate
        #
        # alpha = 0  -> salida = memoria original
        # alpha != 0 -> comienza a incorporar refinamiento
        # ----------------------------------------------------

        output_memory = (
            memory_prompt
            +
            torch.tanh(self.alpha)
            * (refined_memory - memory_prompt)
        )

        # Opcional: devolver información para análisis

        if return_aux:

            return (
                output_memory,
                relation_ids,
                attention_weights
            )

        return output_memory