import json
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple


logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore

try:
    import ollama  # type: ignore
except Exception:
    ollama = None  # type: ignore


@dataclass
class OllamaClient:
    host: str = "http://localhost:11434"

    # Internal reusable clients/sessions for efficiency
    _py_client: Optional[object] = None  # ollama.Client when available
    _session: Optional[object] = None   # requests.Session when requests available

    def _get_py_client(self):
        if ollama is None:
            return None
        if self._py_client is None:
            try:
                self._py_client = ollama.Client(host=self.host)
            except Exception:
                self._py_client = None
        return self._py_client

    def _get_session(self):
        if requests is None:
            return None
        if self._session is None:
            try:
                import requests as _rq  # type: ignore
                self._session = _rq.Session()
            except Exception:
                self._session = None
        return self._session

    def generate(self, model: str, prompt: str, system: Optional[str] = None, temperature: float = 0.7, seed: Optional[int] = None) -> str:
        """Generate non-streamed output using the Ollama Python client if available, else HTTP.
        """
        options = {"temperature": float(temperature)}
        if seed is not None:
            options["seed"] = int(seed)

        last_err: Optional[Exception] = None

        # Preferred path: official ollama Python client
        py_client = self._get_py_client()
        if py_client is not None:
            for attempt in range(1, 4):
                try:
                    logger.debug(f"Ollama(py) generate (attempt {attempt}) model={model}")
                    data = py_client.generate(model=model, prompt=prompt, system=system, options=options)
                    return str(data.get("response", "")).strip()
                except Exception as e:
                    last_err = e
                    wait = min(5.0, 0.5 * (2 ** (attempt - 1)))
                    logger.warning(f"Ollama(py) call failed (attempt {attempt}): {e}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
        # Fallback: HTTP API via requests
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if requests is None:
            raise RuntimeError(
                "Ollama Python client and 'requests' are unavailable. Install 'ollama' or 'requests' to call the API."
            )
        url = self.host.rstrip('/') + "/api/generate"
        session = self._get_session()
        for attempt in range(1, 4):
            try:
                logger.debug(f"Ollama HTTP POST {url} (attempt {attempt}) model={model}")
                if session is not None:
                    resp = session.post(url, json=payload, timeout=60)  # type: ignore
                else:
                    resp = requests.post(url, json=payload, timeout=60)  # type: ignore
                resp.raise_for_status()
                data = resp.json()
                return str(data.get("response", "")).strip()
            except Exception as e:
                last_err = e
                wait = min(5.0, 0.5 * (2 ** (attempt - 1)))
                logger.warning(f"Ollama HTTP call failed (attempt {attempt}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        raise RuntimeError(
            "Ollama call failed after retries. Ensure the server is running and the model is pulled. "
            f"Last error: {last_err}"
        )

    def embed(self, model: str, text: str) -> List[float]:
        """Return a single embedding vector for the text via ollama Python client if available, else HTTP.
        """
        last_err: Optional[Exception] = None
        # Preferred path: official ollama Python client
        py_client = self._get_py_client()
        if py_client is not None:
            for attempt in range(1, 4):
                try:
                    logger.debug(f"Ollama(py) embeddings (attempt {attempt}) model={model} len(text)={len(text)}")
                    data = py_client.embeddings(model=model, input=text)
                    emb = data.get("embedding")
                    if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
                        raise ValueError("Unexpected embeddings response schema from ollama(py)")
                    return [float(x) for x in emb]
                except Exception as e:
                    last_err = e
                    wait = min(5.0, 0.5 * (2 ** (attempt - 1)))
                    logger.warning(f"Ollama(py) embeddings failed (attempt {attempt}): {e}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)
        # Fallback: HTTP API via requests
        if requests is None:
            raise RuntimeError(
                "Ollama Python client and 'requests' are unavailable. Install 'ollama' or 'requests' to call the API."
            )
        url = self.host.rstrip('/') + "/api/embeddings"
        payload = {"model": model, "input": text}
        session = self._get_session()
        for attempt in range(1, 4):
            try:
                logger.debug(f"Ollama HTTP POST {url} (attempt {attempt}) model={model} len(text)={len(text)}")
                if session is not None:
                    resp = session.post(url, json=payload, timeout=60)  # type: ignore
                else:
                    resp = requests.post(url, json=payload, timeout=60)  # type: ignore
                resp.raise_for_status()
                data = resp.json()
                emb = data.get("embedding")
                if not isinstance(emb, list) or not all(isinstance(x, (int, float)) for x in emb):
                    raise ValueError("Unexpected embeddings response schema")
                return [float(x) for x in emb]
            except Exception as e:
                last_err = e
                wait = min(5.0, 0.5 * (2 ** (attempt - 1)))
                logger.warning(f"Ollama embeddings HTTP call failed (attempt {attempt}): {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        raise RuntimeError(
            "Ollama embeddings call failed after retries. Ensure the server is running and the model is pulled. "
            f"Last error: {last_err}"
        )


    def nlp(self, model: str, text: str) -> dict:
        """Perform basic NLP with an Ollama instruct model (e.g., gpt-oss).
        Returns a dict with keys: tokens (list[str]), sentences (list[str]), entities (list[{text,type}]).
        Strict JSON parsing with minimal recovery (extract first {...}).
        """
        system = (
            "You are an NLP annotator. Respond in strict JSON only with keys: "
            "tokens (array of strings), sentences (array of strings), entities (array of objects with 'text' and 'type')."
        )
        prompt = (
            "Text to annotate:\n" + text.strip() + "\n\n"
            "Produce JSON exactly in this schema: {\"tokens\":[...],\"sentences\":[...],\"entities\":[{\"text\":\"...\",\"type\":\"...\"}]}"
        )
        raw = self.generate(model=model, prompt=prompt, system=system, temperature=0.0)
        # Extract JSON object
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Ollama NLP response is not strict JSON.")
        try:
            obj = json.loads(raw[start:end+1])
        except Exception as e:
            raise ValueError("Failed to parse JSON from Ollama NLP response.") from e
        tokens = obj.get("tokens", [])
        sentences = obj.get("sentences", [])
        entities = obj.get("entities", [])
        # Basic validation
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            tokens = []
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            sentences = []
        if not isinstance(entities, list):
            entities = []
        # Normalize entity items
        norm_entities = []
        for e in entities:
            try:
                txt = str(e.get("text", ""))
                typ = str(e.get("type", ""))
                if txt:
                    norm_entities.append({"text": txt, "type": typ})
            except Exception:
                continue
        return {"tokens": tokens, "sentences": sentences, "entities": norm_entities}

    # ---------------------- Prompting and parsing helpers ----------------------

    def parse_label_summary(self, raw: str, allowed_labels: List[str]) -> Tuple[str, str]:
        txt = raw.strip()
        start = txt.find('{')
        end = txt.rfind('}')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Ollama response is not strict JSON with 'label' and 'summary' fields.")
        try:
            obj = json.loads(txt[start:end+1])
        except Exception as e:
            raise ValueError("Failed to parse JSON from Ollama response.") from e
        label = str(obj.get("label", "")).strip()
        summary = str(obj.get("summary", "")).strip()
        if not label or not summary:
            raise ValueError("Ollama JSON must contain non-empty 'label' and 'summary'.")
        if label not in allowed_labels:
            raise ValueError(f"Label '{label}' not in allowed set: {allowed_labels}")
        return label, summary


# Shared client cache per host to avoid re-instantiation
_SHARED_CLIENTS: dict = {}

def get_shared_client(host: str = "http://localhost:11434") -> OllamaClient:
    client = _SHARED_CLIENTS.get(host)
    if client is None:
        client = OllamaClient(host=host)
        _SHARED_CLIENTS[host] = client
    return client


def build_prompt(text: str, context: str, labels: List[str], rigidity: float) -> Tuple[str, str]:
    label_list = ", ".join(labels)
    base_system = (
        "You are an expert annotator. Respond in strict JSON only. "
        "Keys: label (one of [{labels}]) and summary (1-2 sentences)."
    ).format(labels=label_list)
    if context == "C_open":
        system = base_system + " Avoid extra commentary."
        rubric = "Provide your best label and a concise summary."
    elif context == "C_medium":
        system = base_system + " Prefer precise domain labels and keep the summary neutral."
        rubric = (
            "Choose the most specific label matching domain cues. "
            "Summary: what is asserted, avoiding speculation."
        )
    else:  # C_rigid
        system = base_system + " Follow the rubric exactly."
        rubric = (
            "Label selection rubric: If any technology terms appear, choose 'technology'; "
            "biology terms -> 'biology'; finance -> 'finance'; politics -> 'politics'; sports -> 'sports'; art -> 'art'; else 'general'. "
            "Summary rubric: Extract 1 key sentence stating the main claim; avoid opinions."
        )
    prompt = (
        "Text:\n" + text.strip() + "\n\n" +
        rubric + "\n" +
        "Respond as JSON: {\"label\": <label>, \"summary\": <summary>}"
    )
    return prompt, system

