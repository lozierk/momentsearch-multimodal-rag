"""Multimodal LLM — cited answer synthesis from frames, per-tenant switchable.

Every call takes an LLMConfig. Where it comes from (resolved in
src/rag/search.py):
  1. the user's own hosted model (ms_user_llms row — a vLLM/Ollama/LM Studio/
     Together/OpenRouter endpoint via base_url, NVIDIA NIM, or Anthropic), or
  2. the server-wide LLM_* env config as the fallback.

The two multimodal calls are where latency and cost actually live (retrieval
is milliseconds), so frames are downscaled to LLM_IMAGE_MAX_PX before they are
sent and only TOP_K of them ever reach the model.

Providers:
  * "openai"    — Chat Completions; covers every OpenAI-compatible server
                  (vLLM, Ollama, LM Studio, Together, Groq, OpenRouter, ...)
                  via base_url.
  * "nvidia"    — NVIDIA NIM / build.nvidia.com hosted vision models.
                  OpenAI-compatible, same client with NVIDIA's endpoint.
  * "anthropic" — the Anthropic Messages API.

Provider SDKs are imported lazily — only the one you use.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from . import config

# NVIDIA's hosted inference endpoint (OpenAI-compatible).
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

PROVIDERS = ("openai", "nvidia", "anthropic")

SYSTEM = (
    "You answer a user's question about a library of videos and documents using "
    "the numbered moments provided as your evidence. A moment is one of two "
    "things:\n"
    "  * a VIDEO moment, labelled with a timestamp (e.g. @ 01:23). It may include "
    "a video FRAME (what was shown on screen) and/or a TRANSCRIPT excerpt (what "
    "was said out loud).\n"
    "  * a DOCUMENT page, labelled with a page or slide reference (e.g. page "
    "\"p. 4\" or \"slide 7\") from a PDF paper or slide deck. It may include an "
    "IMAGE of that page (the figures, diagrams and layout as printed) and/or the "
    "PAGE TEXT written on it.\n"
    "Use BOTH kinds of evidence: for a question about what someone SAID or wrote, "
    "read the transcript or page text; for a question about what is SHOWN, read "
    "the frame or page image. Refer to a document moment by its page or slide "
    "(\"slide 7 shows…\"), never by a timestamp, and to a video moment by its "
    "timestamp, never by a page.\n"
    "Rules:\n"
    "1. Read the question carefully and answer exactly what is asked. Start with a "
    "one-line direct answer, then explain in short paragraphs — ONE paragraph per "
    "distinct point. Keep it focused, don't pad. No preamble, don't restate the "
    "question.\n"
    "2. Ground every claim in the moments and cite the moment number(s) in square "
    "brackets, e.g. [1] or [2, 3]. When the question is about what was said, quote "
    "the transcript accurately — keep the actual wording and numbers, don't alter "
    "or round them.\n"
    "3. Group the relevant moments by the point they make:\n"
    "   - Moments that make the SAME point (especially several from the same "
    "video) belong TOGETHER in ONE paragraph, cited together, e.g. [1, 2]. Do not "
    "split one shared point across separate paragraphs.\n"
    "   - Moments that make DIFFERENT points, or come from different videos, go in "
    "SEPARATE paragraphs, each with its own citation.\n"
    "   Cover every distinct relevant point — don't merge unrelated ones and don't "
    "drop any.\n"
    "4. Don't use outside knowledge or invent details that aren't in the moments.\n"
    "5. Abstain ONLY as a last resort: if — and only if — none of the moments are "
    "relevant to the question at all, reply with a single sentence saying you "
    "couldn't find it. If even one moment is relevant, ANSWER from it; do not "
    "refuse just because the match is partial."
)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024


def env_config() -> LLMConfig | None:
    """The server-wide fallback model from LLM_* env vars, if configured."""
    if not config.llm_configured():
        return None
    return LLMConfig(provider=config.LLM_PROVIDER, model=config.LLM_MODEL,
                     api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                     max_tokens=config.LLM_MAX_TOKENS)


def from_row(row: dict) -> LLMConfig:
    """A tenant's own hosted model (ms_user_llms row)."""
    return LLMConfig(provider=row.get("provider") or "openai",
                     model=row.get("model") or "",
                     api_key=row.get("api_key") or "",
                     base_url=row.get("base_url") or "",
                     max_tokens=config.LLM_MAX_TOKENS)


def _intro(question: str, n: int) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"Answer this question using the {n} moments below (numbered 1 to {n}). "
        "Each is either a video moment at a timestamp (a frame and/or a "
        "transcript excerpt) or a page of a document at a page/slide reference "
        "(the page image and/or its page text). If the question is about what "
        "was said or written, use the transcript or page text. Give a direct "
        "answer grounded in the relevant moment(s), cited as [n], naming each "
        "one by its timestamp or its page/slide. Only say you couldn't find it "
        "if none of the moments are relevant."
    )


def _downscale(jpeg: bytes) -> bytes:
    """Shrink a frame before it becomes LLM image tokens."""
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg))
    if max(img.size) <= config.LLM_IMAGE_MAX_PX:
        return jpeg
    img.thumbnail((config.LLM_IMAGE_MAX_PX, config.LLM_IMAGE_MAX_PX))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def answer(question: str, moments: list[dict], cfg: LLMConfig) -> str:
    """Synthesize a cited answer from retrieved moments with `cfg`'s model.

    moments: [{"image": bytes|None, "transcript": str|None, "timestamp": str,
    "page": int|None}] — each may carry an image, its text, or both; `page` set
    means it is a document page ("p. 4"/"slide 7"), not a video timestamp."""
    if cfg.provider == "anthropic":
        return _answer_anthropic(cfg, question, moments)
    return _answer_openai(cfg, question, moments)


def ping(cfg: LLMConfig) -> str:
    """Connectivity + vision check: one tiny image, one word back. Raises with
    the provider's error on failure (surfaced to the settings UI)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 40, 40)).save(buf, format="JPEG")
    return answer("Reply with the dominant color of moment 1, one word.",
                  [{"image": buf.getvalue(), "transcript": None, "timestamp": "00:00"}], cfg)


def _meter(model: str, usage, in_field: str, out_field: str) -> None:
    """Record one call's token usage (src/metrics.py). The two providers name
    the same two numbers differently — Anthropic input/output, OpenAI
    prompt/completion — so the caller passes the field names.

    Never raises: a metrics bug must not turn a good answer into a 500, and a
    provider that omits `usage` is a missing datapoint, not a failure."""
    try:
        from . import metrics
        metrics.metrics.record_llm(model,
                                   int(getattr(usage, in_field, 0) or 0),
                                   int(getattr(usage, out_field, 0) or 0))
    except Exception:
        pass


def _base_url(cfg: LLMConfig) -> str | None:
    if cfg.base_url:
        return cfg.base_url
    if cfg.provider == "nvidia":
        return NVIDIA_BASE_URL
    return None


def _label(i: int, m: dict) -> str:
    """One evidence header per moment. A document page says so in words — the
    model must never read "p. 4" as a timestamp (or cite it as one)."""
    doc = m.get("page") is not None
    line = (f"[{i}] document page {m.get('timestamp', '')}" if doc
            else f"[{i}] @ {m.get('timestamp', '')}")
    if m.get("transcript"):
        field = "page text" if doc else "transcript"
        line += f' {field}: "{m["transcript"]}"'
    if m.get("image") is None:
        line += " (page text only, no image)" if doc else " (transcript only, no frame)"
    return line


def _answer_openai(cfg: LLMConfig, question: str, moments: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=_base_url(cfg))
    content: list[dict] = [{"type": "text", "text": _intro(question, len(moments))}]
    for i, m in enumerate(moments, 1):
        content.append({"type": "text", "text": _label(i, m)})
        if m.get("image"):
            uri = f"data:image/jpeg;base64,{base64.b64encode(_downscale(m['image'])).decode()}"
            content.append({"type": "image_url", "image_url": {"url": uri}})
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=cfg.max_tokens,
    )
    _meter(cfg.model, getattr(resp, "usage", None), "prompt_tokens", "completion_tokens")
    return (resp.choices[0].message.content or "").strip()


def _answer_anthropic(cfg: LLMConfig, question: str, moments: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
    blocks: list[dict] = [{"type": "text", "text": _intro(question, len(moments))}]
    for i, m in enumerate(moments, 1):
        blocks.append({"type": "text", "text": _label(i, m)})
        if m.get("image"):
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(_downscale(m["image"])).decode()}})
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": blocks}],
    )
    _meter(cfg.model, getattr(resp, "usage", None), "input_tokens", "output_tokens")
    return "".join(b.text for b in resp.content if b.type == "text").strip()
