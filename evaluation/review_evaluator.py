"""Pure scoring evaluator — no Neo4j, no retrieval logic."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

EMOTION_LABELS   = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
MAX_TEXT_CHARS   = 1800   # ~512 tokens for DistilRoBERTa; safety margin
MIN_REF_CHARS    = 30     # below this, signal-to-noise is too low
MAX_REFERENCES   = 5      # cap for cost/variance control

# Hugging Face hub IDs (used when local paths are unset or missing)
EMOTION_MODEL_HF   = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL_HF = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def _resolve_model_source(env_key: str, hf_default: str) -> str:
    """Prefer EMOTION_MODEL_PATH / SENTIMENT_MODEL_PATH; else download from HF hub."""
    local = (os.environ.get(env_key) or "").strip()
    if local:
        path = Path(local).expanduser()
        if path.is_dir():
            return str(path)
    return hf_default


def emotion_model_source() -> str:
    return _resolve_model_source("EMOTION_MODEL_PATH", EMOTION_MODEL_HF)


def sentiment_model_source() -> str:
    return _resolve_model_source("SENTIMENT_MODEL_PATH", SENTIMENT_MODEL_HF)


def embedding_model_source() -> str:
    """Same env as inference.task_a / pipeline (EMBED_MODEL)."""
    return os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

_emotion_pipeline   = None
_sentiment_pipeline = None
_embedder           = None


def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        from transformers import pipeline
        _emotion_pipeline = pipeline(
            "text-classification",
            model=emotion_model_source(),
            top_k=None, device=-1, truncation=True, max_length=512,
        )
    return _emotion_pipeline


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "text-classification",
            model=sentiment_model_source(),
            top_k=None, device=-1, truncation=True, max_length=512,
        )
    return _sentiment_pipeline


def _get_embedder():
    """Reuses the SentenceTransformer singleton cached by inference.task_a."""
    global _embedder
    if _embedder is None:
        from inference.task_a import get_embedder
        _embedder = get_embedder()
    return _embedder


def warmup_eval_models(show_progress: bool = True) -> None:
    """Load emotion, sentiment, and embedding models (local paths or HF cache)."""
    loaders: list[tuple[str, Callable[[], object]]] = [
        (f"emotion · {emotion_model_source()}", _get_emotion_pipeline),
        (f"sentiment · {sentiment_model_source()}", _get_sentiment_pipeline),
        (f"embedding · {embedding_model_source()}", _get_embedder),
    ]
    iterator = loaders
    if show_progress:
        iterator = tqdm(loaders, desc="Loading eval models", unit="model")
    for _label, loader in iterator:
        loader()


def _normalize_scores(raw: list[dict], all_labels: list[str]) -> dict[str, float]:
    """Convert pipeline output [{'label':'Joy','score':0.9},...] → {'joy':0.9,...}.
    Guarantees all expected labels are present (missing → 0.0), then re-normalizes."""
    out = {lbl: 0.0 for lbl in all_labels}
    for item in raw:
        key = item["label"].lower()
        if key in out:
            out[key] = float(item["score"])
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _prepare_reference_texts(refs: list[str]) -> list[str]:
    """Filter + cap references. Longer refs = stronger signal."""
    cleaned = []
    for r in refs:
        if not r:
            continue
        t = r.strip()[:MAX_TEXT_CHARS]
        if len(t) >= MIN_REF_CHARS:
            cleaned.append(t)
    cleaned.sort(key=len, reverse=True)
    return cleaned[:MAX_REFERENCES]


@dataclass
class EvaluationResult:
    emotional_tone_error:       float
    sentiment_attitude_error:   float
    topic_relevance_error:      float
    combined_score:             float
    predicted_emotion_scores:   dict[str, float]
    avg_reference_emotion:      dict[str, float]
    predicted_sentiment_scores: dict[str, float]
    avg_reference_sentiment:    dict[str, float]
    per_reference_topic_errors: list[float]
    n_references_used:          int
    n_references_filtered_out:  int
    notes: str = ""


class ReviewGenerationEvaluator:
    def evaluate(
        self,
        predicted_text: str,
        reference_texts: list[str],
        *,
        show_progress: bool = True,
    ) -> EvaluationResult:
        pred = predicted_text.strip()[:MAX_TEXT_CHARS]
        refs = _prepare_reference_texts(reference_texts)
        n_filtered_out = len([r for r in reference_texts if r and r.strip()]) - len(refs)

        if not refs:
            return self._null_result(predicted_text, n_filtered_out,
                                     "no valid references after filtering")

        all_texts = [pred] + refs
        score_steps = [
            "emotion classifier",
            "sentiment classifier",
            "topic embeddings",
        ]
        step_iter = score_steps
        if show_progress:
            step_iter = tqdm(score_steps, desc="Scoring review", unit="step", leave=False)

        emo_pipe  = _get_emotion_pipeline()
        sent_pipe = _get_sentiment_pipeline()
        emo_raw, sent_raw, vecs = None, None, None
        for step in step_iter:
            if step == "emotion classifier":
                emo_raw = emo_pipe(all_texts)
            elif step == "sentiment classifier":
                sent_raw = sent_pipe(all_texts)
            elif step == "topic embeddings":
                embedder = _get_embedder()
                vecs = embedder.encode(
                    all_texts, show_progress_bar=False, normalize_embeddings=True
                )

        emo_scores  = [_normalize_scores(x, EMOTION_LABELS)   for x in emo_raw]
        sent_scores = [_normalize_scores(x, SENTIMENT_LABELS) for x in sent_raw]
        pred_emo,  ref_emos   = emo_scores[0],  emo_scores[1:]
        pred_sent, ref_sents  = sent_scores[0], sent_scores[1:]

        pred_vec, ref_vecs = vecs[0], vecs[1:]
        emo_errs   = [self._mae(pred_emo,  re, EMOTION_LABELS)   for re in ref_emos]
        sent_errs  = [self._mae(pred_sent, rs, SENTIMENT_LABELS) for rs in ref_sents]
        topic_errs = [float(1.0 - float(pred_vec @ rv))          for rv in ref_vecs]

        n = len(refs)
        E = sum(emo_errs)   / n
        S = sum(sent_errs)  / n
        T = sum(topic_errs) / n
        score = max(0.0, min(1.0, 1.0 - (0.25 * E + 0.25 * S + 0.5 * T)))

        avg_emo  = {k: sum(r[k] for r in ref_emos)  / n for k in EMOTION_LABELS}
        avg_sent = {k: sum(r[k] for r in ref_sents) / n for k in SENTIMENT_LABELS}

        return EvaluationResult(
            emotional_tone_error=E,
            sentiment_attitude_error=S,
            topic_relevance_error=T,
            combined_score=score,
            predicted_emotion_scores=pred_emo,
            avg_reference_emotion=avg_emo,
            predicted_sentiment_scores=pred_sent,
            avg_reference_sentiment=avg_sent,
            per_reference_topic_errors=topic_errs,
            n_references_used=n,
            n_references_filtered_out=n_filtered_out,
        )

    @staticmethod
    def _mae(a: dict[str, float], b: dict[str, float], labels: list[str]) -> float:
        return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in labels) / len(labels)

    @staticmethod
    def _null_result(predicted_text: str, n_filtered_out: int, note: str) -> EvaluationResult:
        zero_emo  = {k: 0.0 for k in EMOTION_LABELS}
        zero_sent = {k: 0.0 for k in SENTIMENT_LABELS}
        return EvaluationResult(
            emotional_tone_error=0.0,
            sentiment_attitude_error=0.0,
            topic_relevance_error=0.0,
            combined_score=0.0,
            predicted_emotion_scores=zero_emo,
            avg_reference_emotion=zero_emo,
            predicted_sentiment_scores=zero_sent,
            avg_reference_sentiment=zero_sent,
            per_reference_topic_errors=[],
            n_references_used=0,
            n_references_filtered_out=n_filtered_out,
            notes=note,
        )
