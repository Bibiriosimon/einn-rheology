import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class KREPCorrectionModel(nn.Module):
    def __init__(self, scibert_dir, num_dim=4, hidden_dim=256, dropout=0.1, text_pool='cls', delta_bins=5):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(scibert_dir)
        self.text_pool = text_pool
        self.delta_bins = int(delta_bins)
        text_dim = int(self.text_encoder.config.hidden_size)

        self.num_net = nn.Sequential(
            nn.Linear(num_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.evidence_proj = nn.Linear(text_dim, hidden_dim)
        self.summary_proj = nn.Linear(text_dim, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.delta_bin_head = nn.Linear(hidden_dim, self.delta_bins)
        self.delta_refine_head = nn.Linear(hidden_dim, 1)

    def freeze_text(self, freeze_n_layers=10, freeze_embeddings=True):
        if freeze_embeddings:
            for p in self.text_encoder.embeddings.parameters():
                p.requires_grad = False
        layers = getattr(getattr(self.text_encoder, 'encoder', None), 'layer', None)
        if layers is not None:
            for layer in list(layers)[:max(0, freeze_n_layers)]:
                for p in layer.parameters():
                    p.requires_grad = False

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if self.text_pool == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden[:, 0, :]

    def forward(self, x_num, evidence_ids, evidence_mask, summary_ids, summary_mask, bin_centers, bin_half_widths,
                no_gate=False, random_summary_embeds=None):
        h_num = self.num_net(x_num)
        h_evidence = self.evidence_proj(self.encode_text(evidence_ids, evidence_mask))
        if random_summary_embeds is not None:
            h_summary = self.summary_proj(random_summary_embeds)
        else:
            h_summary = self.summary_proj(self.encode_text(summary_ids, summary_mask))

        if no_gate:
            h_mix = (h_evidence + h_summary) / 2.0
            gate_alpha = h_mix.new_ones(h_mix.size(0), 1) * 0.5
        else:
            gate_alpha = torch.sigmoid(self.gate_net(torch.cat([h_num, h_evidence, h_summary], dim=1)))
            h_mix = gate_alpha * h_evidence + (1.0 - gate_alpha) * h_summary
        fused = self.fusion(torch.cat([h_num, h_evidence, h_summary, h_mix], dim=1))

        delta_logits = self.delta_bin_head(fused)
        delta_probs = torch.softmax(delta_logits, dim=1)
        delta_bin = (delta_probs * bin_centers.unsqueeze(0)).sum(dim=1)
        delta_width = (delta_probs * bin_half_widths.unsqueeze(0)).sum(dim=1)
        delta_refine_raw = self.delta_refine_head(fused).squeeze(1)
        delta_refine = torch.tanh(delta_refine_raw) * delta_width
        delta_pred = delta_bin + delta_refine

        return {
            'delta_logits': delta_logits,
            'delta_probs': delta_probs,
            'delta_bin': delta_bin,
            'delta_refine': delta_refine,
            'delta_pred': delta_pred,
            'gate_alpha': gate_alpha.squeeze(1),
            'evidence_norm': torch.norm(h_evidence, dim=1),
            'summary_norm': torch.norm(h_summary, dim=1),
            'fusion_norm': torch.norm(fused, dim=1),
        }
