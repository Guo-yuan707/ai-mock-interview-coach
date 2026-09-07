"""embedding 层(第 7 课):文本 → 稠密向量。可插拔,不让选型卡主干。

设计(讲得出 why):
- 上层(向量库 / 混合检索)只认一个接口:
      embed(texts: list[str]) -> list[list[float]]
- 底层实现可换,三种从优到劣:
    1. SemanticEmbedder:调"真·语义 embedding"API(env 配了 EMBEDDING_* 就用);
    2. OfflineLexicalEmbedder:零依赖的本地"词典稠密向量"(把分词结果哈希成固定维
       向量,不联网也能跑)——它叫 embedding 但不是语义的,诚实标注,只当兜底/基线;
    3. 都没有 → get_embedder 直接抛带指引的错,调用方决定降级为纯词典检索。

红线:这是"向量检索"的演示与对比骨架;有语义 API 时它才真语义。
写文档/面试话术时必须讲清楚当前用的是哪种,不许拿哈希向量冒充语义 embedding。
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Protocol

from .kb import _tokenize  # 复用同一套中文切词,保证和词典基线同源可对比


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _stable_hash_int(term: str, mod: int) -> int:
    """把词映射到 [0,mod):用 md5 而非内置 hash()。

    为什么:内置 hash 对字符串每进程随机加盐 → 同文本两次向量会不同,
    测试/落库都不稳定;md5 全局一致,同词永远同桶。
    """
    digest = hashlib.md5(term.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % mod


def _l2_normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]


@dataclass
class OfflineLexicalEmbedder:
    """本地兜底 embedder:词 → 哈希成 D 维位向量,TF 加权 + L2 归一。

    它是**确定性**的(同文本同向量),用来在没有语义 API / 重库装不上时,
    让"向量检索 + 混合 + rerank + recall 对比"整条链路照常能跑、能测。
    名字里带 Lexical 就是在提醒:别把它当语义向量。
    """
    dim: int = 256
    name: str = "lexical-hash(local,非语义)"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for term in _tokenize(text):
                vec[_stable_hash_int(term, self.dim)] += 1.0
            out.append(_l2_normalize(vec))
        return out


@dataclass
class SemanticEmbedder:
    """真·语义 embedding(OpenAI 兼容协议)。env 配了就用:
    EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL(如 bge-large-zh 之类)。
    没配/调不通时,调用方应降级到 OfflineLexicalEmbedder,别让它抛在主干上。
    """
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    name: str = "semantic-api"
    dim: int = 1024

    def __post_init__(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "SemanticEmbedder 需要 EMBEDDING_API_KEY/EMBEDDING_BASE_URL/EMBEDDING_MODEL。"
                "没配就用 OfflineLexicalEmbedder,或干脆走纯词典检索。")
        from openai import OpenAI   # 延迟 import,没到用时别拖累包
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder() -> Embedder:
    """工厂:能上真语义就上,否则落本地哈希向量兜底(不抛错,由上层决定要不要纯词典)。"""
    base = os.getenv("EMBEDDING_BASE_URL", "").strip()
    key = os.getenv("EMBEDDING_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if key and base and model:
        try:
            return SemanticEmbedder(base_url=base, api_key=key, model=model)
        except Exception:
            pass                       # 配了但用不了 → 落兜底,别让主干崩
    return OfflineLexicalEmbedder()
