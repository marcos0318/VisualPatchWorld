"""Offline VLM scene-graph extraction and cache management."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


PUSHT_VLM_SYSTEM_PROMPT = """You extract structured scene graphs from PushT robot-control frames.
Return only valid JSON. The schema is:
{
  "objects": [
    {"id": "agent", "category": "circular_pusher", "position": {"x": number, "y": number}},
    {"id": "block", "category": "t_block", "position": {"x": number, "y": number}, "angle": number}
  ],
  "relations": []
}
Use stable IDs exactly: agent and block. Coordinates should be image-plane pixel coordinates."""


@dataclass(frozen=True)
class VLMConfig:
    cache_dir: Path
    provider: str = "cache"
    model: str = "Qwen/Qwen3.5-397B-A17B"
    base_url: str = "https://api.deepinfra.com/v1/openai"
    api_key_env: str = "DEEPINFRA_API_KEY"
    allow_missing: bool = False


class VLMSceneGraphExtractor:
    """Loads cached VLM outputs and can optionally populate them via OpenAI-compatible HTTP."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, episode_index: int, step: int) -> Path:
        return self.config.cache_dir / f"pusht_ep{episode_index:06d}_step{step:06d}.json"

    def cache_path_for_key(self, cache_key: str) -> Path:
        return self.config.cache_dir / f"{cache_key}.json"

    @staticmethod
    def cache_key_from_metadata(metadata: Mapping[str, Any]) -> str:
        from .paths import vlm_cache_key

        return vlm_cache_key(metadata)

    def load_or_extract(
        self,
        *,
        episode_index: int,
        step: int,
        image_bytes: Optional[bytes] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        if metadata is not None:
            path = self.cache_path_for_key(self.cache_key_from_metadata(metadata))
        else:
            path = self.cache_path(episode_index, step)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        if self.config.provider == "cache":
            if self.config.allow_missing:
                return {"missing": True, "episode_index": episode_index, "step": step}
            raise FileNotFoundError(
                f"Missing VLM cache for episode={episode_index} step={step}: {path}"
            )

        if self.config.provider == "openai":
            if image_bytes is None:
                raise ValueError("OpenAI VLM extraction requires image bytes")
            payload = self._call_openai_compatible(image_bytes)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return payload

        raise ValueError(f"Unsupported VLM provider: {self.config.provider}")

    def _call_openai_compatible(self, image_bytes: bytes, *, max_retries: int = 8) -> Mapping[str, Any]:
        api_key = os.environ.get(self.config.api_key_env) or os.environ.get("DEEPINFRA_TOKEN")
        if not api_key:
            raise RuntimeError(
                f"{self.config.api_key_env} or DEEPINFRA_TOKEN is required for provider=openai"
            )
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("The `openai` package is required for provider=openai") from exc

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        client = OpenAI(
            api_key=api_key,
            base_url=self._openai_base_url(),
            timeout=180,
        )
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": PUSHT_VLM_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extract the PushT scene graph."},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                                },
                            ],
                        },
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("VLM returned empty completion content")
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                time.sleep(min(60.0, 2.0 ** attempt))
        raise RuntimeError(f"VLM request failed after {max_retries} attempts: {last_error}") from last_error

    def _openai_base_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url[: -len("/chat/completions")]
        return base_url
