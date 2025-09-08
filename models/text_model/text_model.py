import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import ipdb

# Models that use mean pooling
POOL_MODELS = {"TaylorAI/bge-micro-v2"}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LanguageModel(nn.Module):
    def __init__(self, model="distilbert-base-uncased"):
        super(LanguageModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model_name = model
        if "clip" in self.model_name:
            self.model.vision_model = None
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():

            if "clip" in self.model_name:
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)

        if any(model in self.model_name for model in POOL_MODELS):
            sentence_embeddings = mean_pooling(outputs, inputs["attention_mask"])
            sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.qkv_dwconv = nn.Linear(dim * 3, dim * 3, bias=bias)
        self.project_out = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) -> b head c", head=self.num_heads)
        k = rearrange(k, "b (head c) -> b head c", head=self.num_heads)
        v = rearrange(v, "b (head c) -> b head c", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, "b head c -> b (head c)", head=self.num_heads)

        out = self.project_out(out)
        return out


class LMHead(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=64, heads=4, mode="simple"):
        super(LMHead, self).__init__()
        self.embedding = nn.Linear(embedding_dim, hidden_dim)
        if mode == "simple":
            self.TransformerLayers = nn.Sequential(
                TransformerEncoderLayer(hidden_dim, heads),
            )
        elif mode == "plain":
            self.TransformerLayers = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"mode must be 'complex' or 'simple', but got {mode}")

        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, embedding_dim), nn.Sigmoid())

    def forward(self, x):
        emb = self.embedding(x)
        feature = self.TransformerLayers(emb)
        weights = self.fc1(feature)

        return weights * x
