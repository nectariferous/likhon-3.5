import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    q_rot = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k_rot = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    return q_rot, k_rot

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class AdvancedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.rotary_emb = RotaryPositionalEmbedding(self.attention_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        freqs = self.rotary_emb(query_layer)
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, freqs)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class AdvancedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size * 2)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act_fn = SwiGLU()

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states

class AdvancedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = AdvancedAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = AdvancedMLP(config)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_output
        return hidden_states

class AdvancedLikhon35Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([AdvancedBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class AdvancedLikhon35LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = AdvancedLikhon35Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        lm_logits = self.lm_head(transformer_outputs)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return lm_logits, loss

def create_advanced_likhon35_model(vocab_size=50257, n_embd=1024, n_layer=24, n_head=16):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        intermediate_size=4 * n_embd,
        max_position_embeddings=2048,
        layer_norm_epsilon=1e-5,
    )
    return AdvancedLikhon35LMHeadModel(config)
