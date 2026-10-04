import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("notes", "events", "labs", "prescriptions")


class SoftPromptFusion(nn.Module):
    """Maps each modality into the same fixed-size feature space.

    Every modality owns learnable soft-prompt tokens. The prompts cross-attend
    to that modality's encoder states, so notes, events and labs all become
    `num_prompts` tokens living in one unified space.
    """

    def __init__(self, hidden, num_heads=4, num_prompts=8, dropout=0.1):
        super().__init__()
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.randn(num_prompts, hidden) * 0.02)
            for _ in MODALITIES
        ])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(
                hidden, num_heads, dropout=dropout, batch_first=True
            )
            for _ in MODALITIES
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, states, masks):
        out = []
        for i, (h, m) in enumerate(zip(states, masks)):
            q = self.prompts[i].unsqueeze(0).expand(h.size(0), -1, -1)
            a, _ = self.attn[i](q, h, h, key_padding_mask=~m)
            out.append(self.norm(q + a))
        return torch.cat(out, dim=1)


class GraphKnowledgeEncoder(nn.Module):
    """Embeds the two-level CCS -> ICD graph (coarse + fine granularity)."""

    def __init__(self, graph, tokenizer, embed, hidden, max_len=24):
        super().__init__()

        tok = tokenizer(
            graph["nodes"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.register_buffer("node_ids", tok.input_ids)
        self.register_buffer("node_mask", tok.attention_mask)
        self.register_buffer(
            "levels", torch.tensor(graph["levels"], dtype=torch.long)
        )

        n = len(graph["nodes"])
        adj = torch.zeros(n, n)
        for child, parent in graph["edges"]:
            adj[child, parent] = 1.0
            adj[parent, child] = 1.0
        adj = adj / adj.sum(1, keepdim=True).clamp(min=1.0)
        self.register_buffer("adj", adj)

        self.embed = embed
        self.num_coarse = graph["num_coarse"]
        self.level_embedding = nn.Embedding(2, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self):
        m = self.node_mask.unsqueeze(-1).to(self.level_embedding.weight.dtype)
        h = (self.embed(self.node_ids) * m).sum(1) / m.sum(1).clamp(min=1.0)
        h = h + self.level_embedding(self.levels)
        # one round of parent <-> child message passing
        return self.norm(h + F.gelu(self.message(self.adj @ h)))


class KnowledgeAttention(nn.Module):
    """EHR features (queries) attend to knowledge-graph nodes."""

    def __init__(self, hidden, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, ehr, nodes):
        kv = nodes.unsqueeze(0).expand(ehr.size(0), -1, -1)
        attended, weights = self.attn(ehr, kv, kv, need_weights=True)
        return self.norm(ehr + attended), weights


class KnowledgeCalibration(nn.Module):
    """Scores pooled features against the coarse (CCS) node embeddings.

    The code logits are supervised by L_c and their probabilities gate the
    features passed to the generator.
    """

    def __init__(self, hidden, num_coarse, prior_logits=None, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)

        # Patient-specific path: masked mean of the raw encoder states of each
        # modality. The fused soft-prompt features are dominated by the
        # learnable prompts (identical for every patient), so on their own they
        # carry almost no per-patient signal. The final layer starts at zero so
        # training begins exactly at the cosine + prior logits.
        self.direct = nn.Sequential(
            nn.LayerNorm(hidden * len(MODALITIES)),
            nn.Dropout(dropout),
            nn.Linear(hidden * len(MODALITIES), num_coarse),
        )
        nn.init.zeros_(self.direct[-1].weight)
        nn.init.zeros_(self.direct[-1].bias)
        self.gate = nn.Linear(num_coarse, hidden)
        self.num_coarse = num_coarse

        # Cosine similarity with a learnable temperature keeps the logits in
        # a sane range; the bias starts at each category's base rate so the
        # head only has to learn how a patient differs from the average.
        self.scale = nn.Parameter(torch.tensor(5.0))
        bias = (
            prior_logits
            if prior_logits is not None
            else torch.full((num_coarse,), -3.0)
        )
        self.bias = nn.Parameter(bias.clone())

    def forward(self, features, nodes, direct=None):
        pooled = features.mean(1)
        coarse = nodes[: self.num_coarse]
        q = F.normalize(self.query(pooled), dim=-1)
        c = F.normalize(coarse, dim=-1)
        code_logits = self.scale * (q @ c.t()) + self.bias
        if direct is not None:
            code_logits = code_logits + self.direct(direct)
        gate = torch.sigmoid(self.gate(torch.sigmoid(code_logits)))
        return features * (1.0 + gate.unsqueeze(1)), code_logits


class KnowGenT5(nn.Module):
    """EHR-KnowGen: multimodal soft-prompt fusion + graph knowledge + LLM.

    notes / events / labs / prescriptions
      -> shared T5 encoder (per modality)
      -> SoftPromptFusion            (unified feature space)
      -> Knowledge Attention <-> Knowledge Calibration (L_c)
         over the external CCS/ICD knowledge graph
      -> T5 decoder generates diagnoses (L_f)

    total loss = L_f + calib_weight * L_c
    """

    def __init__(
        self,
        graph,
        model_name="t5-small",
        num_heads=4,
        num_prompts=8,
        dropout=0.1,
        calib_weight=0.5,
        class_prior=None,
        pos_weight=1.0,
    ):
        super().__init__()

        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        hidden = self.t5.config.d_model

        self.graph = graph
        self.code_vocab = list(graph["coarse_names"])
        self.code_to_idx = {c: i for i, c in enumerate(self.code_vocab)}
        self.calib_weight = calib_weight
        self.pos_weight = pos_weight

        self.fusion = SoftPromptFusion(hidden, num_heads, num_prompts, dropout)
        self.knowledge = GraphKnowledgeEncoder(
            graph, self.tokenizer, self.t5.shared, hidden
        )
        self.knowledge_attention = KnowledgeAttention(
            hidden, num_heads, dropout
        )
        prior_logits = None
        if class_prior:
            p = torch.tensor([
                min(max(class_prior.get(c, 0.0), 0.002), 0.5)
                for c in self.code_vocab
            ])
            prior_logits = torch.log(p / (1 - p))

        self.knowledge_calibration = KnowledgeCalibration(
            hidden, graph["num_coarse"], prior_logits, dropout
        )

    # ---------------------------------------------------------------
    def encode(self, batch, max_length=192):
        device = next(self.parameters()).device

        states, masks, pooled = [], [], []
        for name in MODALITIES:
            texts = [t if t else "none" for t in batch[name]]
            tok = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            h = self.t5.encoder(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
            ).last_hidden_state
            states.append(h)
            masks.append(tok.attention_mask.bool())

            m = tok.attention_mask.unsqueeze(-1).to(h.dtype)
            pooled.append((h * m).sum(1) / m.sum(1).clamp(min=1.0))

        unified = self.fusion(states, masks)
        nodes = self.knowledge()

        attended, weights = self.knowledge_attention(unified, nodes)
        calibrated, code_logits = self.knowledge_calibration(
            attended, nodes, direct=torch.cat(pooled, dim=-1)
        )
        return calibrated, code_logits, weights

    def _code_targets(self, target_lists, device):
        t = torch.zeros(len(target_lists), len(self.code_vocab), device=device)
        for i, codes in enumerate(target_lists):
            for c in codes:
                j = self.code_to_idx.get(str(c))
                if j is not None:
                    t[i, j] = 1.0
        return t

    # ---------------------------------------------------------------
    def forward(
        self,
        batch,
        target_texts,
        target_lists,
        max_length=192,
        target_max_length=256,
    ):
        from transformers.modeling_outputs import BaseModelOutput

        features, code_logits, _ = self.encode(batch, max_length)
        device = features.device

        labels = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=target_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        labels[labels == self.tokenizer.pad_token_id] = -100

        out = self.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=features),
            labels=labels,
        )

        loss_f = out.loss
        loss_c = F.binary_cross_entropy_with_logits(
            code_logits,
            self._code_targets(target_lists, device),
            pos_weight=torch.full(
                (len(self.code_vocab),), self.pos_weight, device=device
            ),
        )

        return {
            "loss": loss_f + self.calib_weight * loss_c,
            "loss_f": loss_f,
            "loss_c": loss_c,
            "code_logits": code_logits,
        }

    @torch.no_grad()
    def generate(self, batch, max_length=256, num_beams=1):
        from transformers.modeling_outputs import BaseModelOutput

        features, _, _ = self.encode(batch)
        ids = self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=features),
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=4,
        )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def predict_codes(self, batch, threshold=0.5):
        """Detect diseases: CCS categories whose calibrated score passes
        the threshold (the knowledge-calibration head)."""

        _, code_logits, _ = self.encode(batch)
        probs = torch.sigmoid(code_logits)
        return probs, [
            {self.code_vocab[j] for j in row.nonzero().flatten().tolist()}
            for row in (probs > threshold)
        ]

    @torch.no_grad()
    def explain(self, batch, top_k=5, max_length=192):
        """Evidence for each patient: the knowledge-graph nodes the model
        attended to most, and the CCS categories it scored highest."""

        _, code_logits, weights = self.encode(batch, max_length)
        node_score = weights.mean(1)  # [B, nodes]
        probs = torch.sigmoid(code_logits)

        result = []
        for b in range(node_score.size(0)):
            nodes = node_score[b].topk(top_k).indices.tolist()
            codes = probs[b].topk(top_k)
            result.append({
                "attended_nodes": [
                    (self.graph["nodes"][i], round(node_score[b, i].item(), 4))
                    for i in nodes
                ],
                "top_categories": [
                    (self.code_vocab[i], round(p, 4))
                    for p, i in zip(codes.values.tolist(),
                                    codes.indices.tolist())
                ],
            })
        return result
