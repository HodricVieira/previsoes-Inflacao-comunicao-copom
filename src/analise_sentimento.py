"""Coleta e analisa o tom de atas e comunicados oficiais do Copom."""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import os
import random
import re
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
BCB_COPOM_API = "https://www.bcb.gov.br/api/servico/sitebcb/copom"
DEFAULT_START_DATE = "2012-01-01"
CHUNK_SIZE = 3500
MIN_MANUAL_LABELS = 200
PROMPT_VERSION = "copom-tone-v1"
VALID_LABELS = {"expansionista", "neutro", "contracionista"}
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
    "gemini-3-flash-preview",
]


class DailyQuotaExceededError(RuntimeError):
    """Indica esgotamento da cota diária (RPD) do modelo na API."""

LEXICON = {
    "contracionista": {
        "aperto",
        "apertos",
        "contracionista",
        "contracionistas",
        "elevacao",
        "elevacoes",
        "elevar",
        "elevada",
        "elevado",
        "elevados",
        "elevaram",
        "alta",
        "altas",
        "aumento",
        "aumentos",
        "aumentar",
        "aumentou",
        "aumentaram",
        "subir",
        "subida",
        "restritiva",
        "restritivo",
        "restritivas",
        "restritivos",
        "pressao",
        "pressao inflacionaria",
        "pressao de demanda",
        "desancoragem",
        "desancoradas",
        "persistente",
        "persistentes",
        "aceleracao",
    },
    "expansionista": {
        "acomodativa",
        "acomodativo",
        "acomodaticia",
        "afrouxamento",
        "corte",
        "cortes",
        "cortar",
        "cortes de juros",
        "expansionista",
        "expansionistas",
        "flexibilizacao",
        "reducao",
        "reducoes",
        "reduzir",
        "reduziu",
        "reduzida",
        "reduzidas",
        "reduzido",
        "reduzidos",
        "queda",
        "quedas",
        "cair",
        "baixa",
        "baixo",
        "menor",
        "menores",
        "afrouxar",
        "desinflacao",
        "desinflacionario",
        "desinflacionaria",
        "estimulo",
        "estimulativa",
    },
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_tags.append(tag)
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._ignored_tags:
            self._ignored_tags.remove(tag)
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_tags:
            self.parts.append(data)


def _strip_html(raw_text: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw_text)
    text = html.unescape(" ".join(parser.parts))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _api_get(session: requests.Session, path: str, **params: Any) -> dict[str, Any]:
    response = session.get(
        f"{BCB_COPOM_API}/{path}",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not isinstance(result.get("conteudo"), list):
        raise ValueError(f"Resposta inesperada da API do Copom ({path}).")
    return result


def coletar_documentos(start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    """Baixa o texto integral das atas e comunicados pela API aberta do BCB."""
    start = pd.Timestamp(start_date)
    if pd.isna(start):
        raise ValueError(f"Data inicial inválida: {start_date!r}")

    session = requests.Session()
    session.headers.update({"User-Agent": "projeto-ipca-copom/1.0"})
    lists = [
        ("ata", "atas", "atas_detalhes", "dataPublicacao", "nroReuniao"),
        ("comunicado", "comunicados", "comunicados_detalhes", "dataReferencia", "nro_reuniao"),
    ]
    documents: list[dict[str, Any]] = []

    for doc_type, list_endpoint, detail_endpoint, date_key, number_key in lists:
        max_items = 5000 if doc_type == "ata" else 500
        summaries = _api_get(session, list_endpoint, quantidade=max_items)["conteudo"]
        for summary in summaries:
            publication_date = summary.get(date_key)
            if not publication_date or pd.Timestamp(publication_date) < start:
                continue
            meeting_number = summary.get(number_key)
            if meeting_number is None:
                raise ValueError(
                    f"API do BCB sem número da reunião em {doc_type}: {summary}"
                )
            detail = _api_get(
                session, detail_endpoint, nro_reuniao=int(meeting_number)
            )["conteudo"]
            if len(detail) != 1:
                raise ValueError(
                    f"Esperado um detalhe para reunião {meeting_number}, "
                    f"recebidos {len(detail)}."
                )
            item = detail[0]
            raw_text = item.get("textoAta" if doc_type == "ata" else "textoComunicado")
            source_url = (
                f"{BCB_COPOM_API}/{detail_endpoint}"
                f"?nro_reuniao={int(meeting_number)}"
            )
            if not raw_text and doc_type == "ata" and item.get("urlPdfAta"):
                pdf_response = session.get(item["urlPdfAta"], timeout=120)
                pdf_response.raise_for_status()
                pdf = PdfReader(io.BytesIO(pdf_response.content))
                raw_text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
                source_url = item["urlPdfAta"]
            text = _strip_html(str(raw_text or ""))
            if len(text) < 100:
                raise ValueError(
                    f"Texto vazio ou incompleto para {doc_type} "
                    f"da reunião {meeting_number}."
                )
            documents.append(
                {
                    "documento_id": f"{doc_type}_{int(meeting_number):03d}",
                    "tipo_fonte": doc_type,
                    "nro_reuniao": int(meeting_number),
                    "data_publicacao": pd.Timestamp(publication_date),
                    "data_referencia": pd.Timestamp(
                        item.get("dataReferencia", publication_date)
                    ),
                    "titulo": item.get("titulo", summary.get("titulo", "")),
                    "url_fonte": source_url,
                    "texto": text,
                }
            )
            time.sleep(0.02)

    documents_frame = pd.DataFrame(documents)
    if documents_frame.empty:
        raise ValueError(
            f"A API do BCB não retornou documentos desde {start:%Y-%m-%d}."
        )
    if documents_frame["documento_id"].duplicated().any():
        raise ValueError("A API do BCB retornou documentos duplicados.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "documentos_copom.csv"
    documents_frame.sort_values(["data_publicacao", "documento_id"]).to_csv(
        output, index=False, encoding="utf-8-sig"
    )
    print(
        f"Coletados {len(documents_frame)} documentos "
        f"({documents_frame['tipo_fonte'].value_counts().to_dict()}) "
        f"desde {start:%Y-%m}; arquivo: {output}."
    )
    return documents_frame


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _label_for_score(score: float) -> str:
    if score < -0.15:
        return "expansionista"
    if score > 0.15:
        return "contracionista"
    return "neutro"


def _lexicon_score(text: str) -> tuple[float, int, str]:
    normalized = _normalize(text)
    counts: dict[str, int] = {}
    for label, terms in LEXICON.items():
        counts[label] = sum(
            len(re.findall(rf"(?<!\w){re.escape(term)}(?!\w)", normalized))
            for term in terms
        )
    total = sum(counts.values())
    score = (
        (counts["contracionista"] - counts["expansionista"]) / total
        if total
        else 0.0
    )
    return score, total, _label_for_score(score)


def _make_chunks(text: str, max_chars: int = CHUNK_SIZE) -> list[str]:
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            cut = paragraph.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        candidate = f"{current}\n{paragraph}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def preparar_trechos() -> pd.DataFrame:
    """Cria trechos para classificação e uma amostra manual sem sobrescrever rótulos."""
    documents_path = DATA_DIR / "documentos_copom.csv"
    if not documents_path.exists():
        raise FileNotFoundError(
            "Documentos não encontrados. Execute: "
            "python src\\analise_sentimento.py coletar"
        )
    documents = pd.read_csv(
        documents_path,
        encoding="utf-8-sig",
        parse_dates=["data_publicacao", "data_referencia"],
    )
    records: list[dict[str, Any]] = []
    for row in documents.itertuples(index=False):
        chunks = _make_chunks(row.texto)
        for index, chunk in enumerate(chunks, start=1):
            lex_score, lex_hits, lex_label = _lexicon_score(chunk)
            records.append(
                {
                    "trecho_id": f"{row.documento_id}_p{index:03d}",
                    "documento_id": row.documento_id,
                    "tipo_fonte": row.tipo_fonte,
                    "nro_reuniao": row.nro_reuniao,
                    "data_publicacao": row.data_publicacao,
                    "data_referencia": row.data_referencia,
                    "url_fonte": row.url_fonte,
                    "trecho_texto": chunk,
                    "escore_lexico": lex_score,
                    "n_termos_lexico": lex_hits,
                    "rotulo_lexico": lex_label,
                    "escore_llm": np.nan,
                    "confianca_llm": np.nan,
                    "justificativa_llm": "",
                    "rotulo_llm": "",
                    "modelo_llm": "",
                    "versao_prompt": "",
                }
            )

    chunks_frame = pd.DataFrame(records)
    chunks_path = DATA_DIR / "trechos_sentimento.csv"
    if chunks_path.exists():
        old = pd.read_csv(chunks_path, encoding="utf-8-sig")
        preserved_columns = [
            "trecho_id",
            "escore_llm",
            "confianca_llm",
            "justificativa_llm",
            "rotulo_llm",
            "modelo_llm",
            "versao_prompt",
        ]
        old = old[[c for c in preserved_columns if c in old.columns]].drop_duplicates(
            "trecho_id"
        )
        chunks_frame = chunks_frame.drop(
            columns=[c for c in preserved_columns if c != "trecho_id"]
        ).merge(old, on="trecho_id", how="left", validate="one_to_one")
        for column in preserved_columns[1:]:
            if column not in chunks_frame:
                chunks_frame[column] = "" if column in {"justificativa_llm", "rotulo_llm", "modelo_llm"} else np.nan
            elif column in {
                "justificativa_llm",
                "rotulo_llm",
                "modelo_llm",
                "versao_prompt",
            }:
                chunks_frame[column] = (
                    chunks_frame[column].astype("string").fillna("")
                )
    chunks_frame.to_csv(chunks_path, index=False, encoding="utf-8-sig")

    manual_path = DATA_DIR / "rotulacao_manual.csv"
    if manual_path.exists():
        previous = pd.read_csv(manual_path, encoding="utf-8-sig")
        labels = [
            column
            for column in ("rotulo_manual", "rotulo_manual_2")
            if column in previous
        ]
        has_manual_work = any(
            previous[column].astype("string").fillna("").str.strip().ne("").any()
            for column in labels
        )
        has_valid_sample = (
            previous["documento_id"].nunique() >= MIN_MANUAL_LABELS
            and len(previous) >= MIN_MANUAL_LABELS
        )
    else:
        previous = pd.DataFrame()
        labels = []
        has_manual_work = False
        has_valid_sample = False

    if manual_path.exists() and (has_manual_work or has_valid_sample):
        selected_ids = previous[["trecho_id", *labels]].drop_duplicates("trecho_id")
        sample = selected_ids.merge(
            chunks_frame, on="trecho_id", how="left", validate="one_to_one"
        )
    else:
        rng = random.Random(42)
        document_ids = sorted(chunks_frame["documento_id"].unique().tolist())
        chosen_documents = rng.sample(
            document_ids, k=min(MIN_MANUAL_LABELS, len(document_ids))
        )
        chosen = [
            rng.choice(
                chunks_frame.loc[
                    chunks_frame["documento_id"] == document_id, "trecho_id"
                ].tolist()
            )
            for document_id in chosen_documents
        ]
        sample = chunks_frame.loc[chunks_frame["trecho_id"].isin(chosen)].copy()
        sample["rotulo_manual"] = ""
        sample["rotulo_manual_2"] = ""
    for column in ("rotulo_manual", "rotulo_manual_2"):
        if column not in sample:
            sample[column] = ""
    sample = sample[
        [
            "trecho_id",
            "documento_id",
            "tipo_fonte",
            "nro_reuniao",
            "data_publicacao",
            "url_fonte",
            "trecho_texto",
            "rotulo_manual",
            "rotulo_manual_2",
        ]
    ]
    sample.to_csv(manual_path, index=False, encoding="utf-8-sig")
    print(
        f"Preparados {len(chunks_frame)} trechos e amostra manual com "
        f"{len(sample)} linhas em {manual_path}. "
        "Rotule como expansionista, neutro ou contracionista."
    )
    if len(sample) < MIN_MANUAL_LABELS:
        print(
            f"AVISO: a amostra ainda tem menos de {MIN_MANUAL_LABELS} trechos; "
            "a coleta de mais textos/trechos é necessária para cumprir a validação."
        )
    return chunks_frame


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _llm_configuration() -> tuple[str, str, str]:
    _load_dotenv()
    base_url = os.getenv("COPOM_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("COPOM_LLM_API_KEY", "")
    model = os.getenv("COPOM_LLM_MODEL", "gpt-4o-mini")
    hostname = urlparse(base_url).hostname
    if not api_key and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            "Defina COPOM_LLM_API_KEY e, opcionalmente, COPOM_LLM_BASE_URL e "
            "COPOM_LLM_MODEL. Para um servidor local compatível, informe sua URL."
        )
    return base_url, api_key, model


def _extract_json_payload(content: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_retry_delay(
    response: requests.Response | None, attempt: int, model: str = ""
) -> float:
    if response is not None:
        body = response.text or ""
        if response.status_code == 429 and (
            "PerDay" in body or "requests per day" in body.lower()
        ):
            raise DailyQuotaExceededError(
                f"Cota diária (RPD) esgotada para o modelo {model or 'configurado'}."
            )
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay_val = float(retry_after)
                if delay_val > 180.0:
                    raise DailyQuotaExceededError(
                        f"Tempo de espera muito longo ({delay_val:.0f}s) para {model}: cota esgotada."
                    )
                return max(delay_val + 1.0, 5.0)
            except ValueError:
                pass
        match = re.search(
            r"(?:retryDelay[\"']?\s*:\s*[\"']|retry in )([0-9]+(?:\.[0-9]+)?)s",
            body,
            flags=re.IGNORECASE,
        )
        if match:
            delay_val = float(match.group(1))
            if delay_val > 180.0:
                raise DailyQuotaExceededError(
                    f"Tempo de espera muito longo ({delay_val:.0f}s) para {model}: cota esgotada."
                )
            return max(delay_val + 2.0, 5.0)
        if response.status_code == 429:
            return 60.0
    return float(2 ** (attempt + 1))


def _post_chat_json(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    max_attempts: int = 6,
) -> Any:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    messages = (
        [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]
        if model.lower().startswith("gemma")
        else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": messages,
    }
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(15, 180),
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"{response.status_code} {response.reason} para {response.url} "
                    f"(modelo={model}): {response.text}",
                    response=response,
                )
            if not response.ok:
                raise requests.HTTPError(
                    f"{response.status_code} {response.reason} para {response.url} "
                    f"(modelo={model}): {response.text}",
                    response=response,
                )
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Resposta do LLM sem conteúdo textual.")
            return _extract_json_payload(content)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in {429, 500, 502, 503, 504} or attempt == max_attempts - 1:
                raise
            last_exc = exc
            delay = _extract_retry_delay(exc.response, attempt, model=model)
            print(f"HTTP {status}; aguardando {delay:.1f}s antes de retentar...", flush=True)
            time.sleep(delay)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(2 ** min(attempt + 1, 4))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _validate_tone_item(raw: dict[str, Any]) -> tuple[float, float, str, str]:
    score = float(raw["escore"])
    confidence = float(raw["confianca"])
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError(f"Escore LLM fora de [-1, 1]: {score}")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"Confiança LLM fora de [0, 1]: {confidence}")
    return (
        score,
        confidence,
        str(raw.get("justificativa", "")).strip(),
        _label_for_score(score),
    )


def _classify_with_llm(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
) -> tuple[float, float, str, str]:
    system_prompt = (
        "Você analisa comunicação de política monetária do Banco Central do Brasil. "
        "Classifique o TOM da decisão/viés monetário, não se a economia está "
        "bem ou mal. Escala: -1 = expansionista (sinal de flexibilização ou "
        "juros menores); 0 = neutro/equilibrado; +1 = contracionista (sinal de "
        "aperto ou juros maiores). Considere o contexto e não classifique uma "
        "mera descrição de inflação como tom se não houver implicação de política. "
        "Responda apenas JSON com escore numérico entre -1 e 1, confianca entre "
        "0 e 1 e justificativa curta em português. Versão do protocolo: "
        f"{PROMPT_VERSION}."
    )
    result = _post_chat_json(session, base_url, api_key, model, system_prompt, text)
    if not isinstance(result, dict):
        raise ValueError("Resposta JSON do LLM deve ser um objeto.")
    return _validate_tone_item(result)


def _classify_batch_with_llm(
    session: requests.Session,
    base_url: str,
    api_key: str,
    model: str,
    items: list[tuple[str, str]],
) -> dict[str, tuple[float, float, str, str]]:
    """Classifica um lote de trechos em uma única chamada JSON mantendo escore individual."""
    system_prompt = (
        "Você analisa comunicação de política monetária do Banco Central do Brasil. "
        "Para CADA trecho informado abaixo, classifique o TOM da decisão/viés monetário, "
        "não se a economia está bem ou mal. Escala: -1 = expansionista (flexibilização/"
        "juros menores); 0 = neutro/equilibrado; +1 = contracionista (aperto/juros maiores). "
        "Considere o contexto e não classifique uma mera descrição de inflação como tom "
        "se não houver implicação de política. Responda EXCLUSIVAMENTE um array JSON, "
        "contendo exatamente um objeto por trecho com as chaves: "
        '"trecho_id" (id exato informado), "escore" (número entre -1 e 1), '
        '"confianca" (número entre 0 e 1) e "justificativa" (curta em português). '
        f"Versão do protocolo: {PROMPT_VERSION}."
    )
    user_content = "\n\n".join(
        f"=== TRECHO {tid} ===\n{txt}" for tid, txt in items
    )
    parsed = _post_chat_json(session, base_url, api_key, model, system_prompt, user_content)
    if isinstance(parsed, dict):
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if not list_values:
            raise ValueError("Objeto JSON retornado pelo lote não contém lista de resultados.")
        parsed = list_values[0]
    if not isinstance(parsed, list):
        raise ValueError("Resposta em lote deve ser um array JSON.")

    expected_ids = {tid for tid, _ in items}
    updates: dict[str, tuple[float, float, str, str]] = {}
    for entry in parsed:
        if not isinstance(entry, dict) or "trecho_id" not in entry:
            continue
        tid = str(entry["trecho_id"]).strip()
        if tid not in expected_ids:
            continue
        try:
            updates[tid] = _validate_tone_item(entry)
        except (KeyError, TypeError, ValueError):
            continue
    return updates


def classificar_trechos(
    batch_size: int = 25,
    min_interval_seconds: float = 4.5,
    model_override: str | None = None,
) -> pd.DataFrame:
    """Classifica todos os trechos pendentes em lotes respeitando limite de taxa e cota."""
    base_url, api_key, configured_model = _llm_configuration()
    primary_model = model_override or configured_model
    candidate_models = [primary_model]
    if (
        urlparse(base_url).hostname == "generativelanguage.googleapis.com"
        and model_override is None
    ):
        for candidate in GEMINI_FALLBACK_MODELS:
            if candidate not in candidate_models:
                candidate_models.append(candidate)

    chunks = preparar_trechos()
    for col in ("justificativa_llm", "rotulo_llm", "modelo_llm", "versao_prompt"):
        chunks[col] = chunks[col].astype("string").fillna("")

    id_to_index = {str(tid): idx for idx, tid in chunks["trecho_id"].items()}
    session = requests.Session()
    last_request_at = 0.0
    model_idx = 0

    def _wait_rate_limit() -> None:
        nonlocal last_request_at
        if last_request_at > 0.0 and min_interval_seconds > 0.0:
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval_seconds:
                time.sleep(min_interval_seconds - elapsed)
        last_request_at = time.monotonic()

    def _call_with_model_fallback(caller):
        nonlocal model_idx
        while model_idx < len(candidate_models):
            active_model = candidate_models[model_idx]
            try:
                _wait_rate_limit()
                return active_model, caller(active_model)
            except DailyQuotaExceededError as exc:
                print(f"{exc} Tentando próximo modelo...", flush=True)
                model_idx += 1
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in {400, 403, 404} and model_idx + 1 < len(candidate_models):
                    print(
                        f"Modelo {active_model} indisponível (HTTP {status}). Tentando próximo modelo...",
                        flush=True,
                    )
                    model_idx += 1
                else:
                    raise
        raise DailyQuotaExceededError(
            f"Todos os modelos avaliados esgotaram a cota diária (RPD): {candidate_models}"
        )

    def _apply_and_save(
        updates: dict[str, tuple[float, float, str, str]], used_model: str
    ) -> int:
        applied = 0
        for tid, res in updates.items():
            idx = id_to_index.get(tid)
            if idx is None:
                continue
            chunks.loc[
                idx,
                [
                    "escore_llm",
                    "confianca_llm",
                    "justificativa_llm",
                    "rotulo_llm",
                    "modelo_llm",
                    "versao_prompt",
                ],
            ] = [res[0], res[1], res[2], res[3], used_model, PROMPT_VERSION]
            applied += 1
        chunks.to_csv(
            DATA_DIR / "trechos_sentimento.csv", index=False, encoding="utf-8-sig"
        )
        return applied

    def _pending_indices() -> list[int]:
        mask = pd.to_numeric(chunks["escore_llm"], errors="coerce").isna()
        return chunks.index[mask].tolist()

    last_used_model = primary_model
    for current_batch_size in (max(1, int(batch_size)), 5):
        pending = _pending_indices()
        if not pending:
            break
        total_batches = math.ceil(len(pending) / current_batch_size)
        for batch_num, start in enumerate(range(0, len(pending), current_batch_size), start=1):
            batch_indices = pending[start : start + current_batch_size]
            items = [
                (str(chunks.at[idx, "trecho_id"]), str(chunks.at[idx, "trecho_texto"]))
                for idx in batch_indices
            ]
            used_model, updates = _call_with_model_fallback(
                lambda m: _classify_batch_with_llm(session, base_url, api_key, m, items)
            )
            last_used_model = used_model
            n_applied = _apply_and_save(updates, used_model)
            total_done = int(pd.to_numeric(chunks["escore_llm"], errors="coerce").notna().sum())
            print(
                f"Lote {batch_num}/{total_batches} ({used_model}, tamanho {len(items)}): "
                f"{n_applied} classificados | total {total_done}/{len(chunks)}.",
                flush=True,
            )

    for idx in _pending_indices():
        tid = str(chunks.at[idx, "trecho_id"])
        used_model, res = _call_with_model_fallback(
            lambda m: _classify_with_llm(
                session, base_url, api_key, m, str(chunks.at[idx, "trecho_texto"])
            )
        )
        last_used_model = used_model
        _apply_and_save({tid: res}, used_model)
        total_done = int(pd.to_numeric(chunks["escore_llm"], errors="coerce").notna().sum())
        print(
            f"Fallback individual {tid} ({used_model}): total {total_done}/{len(chunks)}.",
            flush=True,
        )

    _assert_all_chunks_classified(chunks)
    print(
        f"Classificação integral concluída ({len(chunks)} trechos) com o modelo {last_used_model}."
    )
    return chunks


def _assert_all_chunks_classified(
    chunks: pd.DataFrame, documents: pd.DataFrame | None = None
) -> None:
    """Exige que 100% dos trechos de todos os documentos estejam classificados pela LLM."""
    if chunks.empty:
        raise ValueError("trechos_sentimento.csv está vazio. Execute coletar e preparar.")
    scores = pd.to_numeric(chunks["escore_llm"], errors="coerce")
    labels = chunks["rotulo_llm"].astype("string").fillna("").str.strip().str.lower()
    invalid_mask = scores.isna() | ~labels.isin(VALID_LABELS)
    if invalid_mask.any():
        n_missing = int(invalid_mask.sum())
        n_docs = int(chunks.loc[invalid_mask, "documento_id"].nunique())
        raise ValueError(
            f"Classificação LLM incompleta: {n_missing}/{len(chunks)} trechos em "
            f"{n_docs} documentos ainda não foram classificados. "
            "Execute `python src\\analise_sentimento.py classificar` antes de agregar ou validar."
        )
    if documents is not None and not documents.empty:
        missing_docs = set(documents["documento_id"]) - set(chunks["documento_id"])
        if missing_docs:
            raise ValueError(
                f"{len(missing_docs)} documentos de documentos_copom.csv não possuem trechos "
                "em trechos_sentimento.csv. Execute `preparar` e `classificar`."
            )


def importar_rotulos_json(
    json_path: Path | None = None,
    model_name: str = "llm_externo_json",
) -> pd.DataFrame:
    """Importa rótulos pré-classificados de um arquivo JSON para trechos_sentimento.csv."""
    source_path = json_path or (DATA_DIR / "rotulo_llm.json")
    if not source_path.exists():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {source_path}")
    raw_items = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("O JSON deve ser uma lista não vazia de objetos.")

    chunks = preparar_trechos()
    valid_ids = set(chunks["trecho_id"])
    updates: dict[str, tuple[float, float, str, str]] = {}
    for item in raw_items:
        trecho_id = str(item["trecho_id"]).strip()
        if trecho_id not in valid_ids:
            raise ValueError(f"trecho_id desconhecido no JSON: {trecho_id}")
        score = float(item["escore"])
        confidence = float(item["confianca"])
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError(f"Escore fora de [-1, 1] para {trecho_id}: {score}")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Confiança fora de [0, 1] para {trecho_id}: {confidence}")
        justification = str(item.get("justificativa", "")).strip()
        updates[trecho_id] = (score, confidence, justification, _label_for_score(score))

    matched_mask = chunks["trecho_id"].isin(updates)
    for idx in chunks.index[matched_mask]:
        tid = str(chunks.at[idx, "trecho_id"])
        score, conf, just, label = updates[tid]
        chunks.loc[
            idx,
            [
                "escore_llm",
                "confianca_llm",
                "justificativa_llm",
                "rotulo_llm",
                "modelo_llm",
                "versao_prompt",
            ],
        ] = [score, conf, just, label, model_name, PROMPT_VERSION]

    chunks.to_csv(
        DATA_DIR / "trechos_sentimento.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"Importados {int(matched_mask.sum())} rótulos de {source_path.name} "
        f"para trechos_sentimento.csv."
    )
    return chunks


def agregar_sentimento() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega escores por documento e mês de publicação exigindo classificação integral."""
    chunks_path = DATA_DIR / "trechos_sentimento.csv"
    documents_path = DATA_DIR / "documentos_copom.csv"
    if not chunks_path.exists() or not documents_path.exists():
        raise FileNotFoundError("Execute os comandos coletar e preparar primeiro.")
    chunks = pd.read_csv(chunks_path, encoding="utf-8-sig")
    documents = pd.read_csv(
        documents_path, encoding="utf-8-sig", parse_dates=["data_publicacao"]
    )
    _assert_all_chunks_classified(chunks, documents=documents)
    for column in ("escore_llm", "escore_lexico"):
        chunks[column] = pd.to_numeric(chunks[column], errors="coerce")

    document_scores = (
        chunks.groupby("documento_id", as_index=False)
        .agg(
            escore_llm=("escore_llm", "mean"),
            n_trechos_llm=("escore_llm", "count"),
            escore_lexico=("escore_lexico", "mean"),
            n_trechos=("trecho_id", "size"),
        )
        .merge(
            documents[
                [
                    "documento_id",
                    "tipo_fonte",
                    "data_publicacao",
                    "nro_reuniao",
                    "titulo",
                    "url_fonte",
                ]
            ],
            on="documento_id",
            how="left",
            validate="one_to_one",
        )
    )
    document_scores.loc[
        document_scores["n_trechos_llm"] != document_scores["n_trechos"],
        "escore_llm",
    ] = np.nan
    document_scores.to_csv(
        DATA_DIR / "sentimento_documentos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    document_scores["data"] = (
        document_scores["data_publicacao"].dt.to_period("M").dt.to_timestamp()
    )
    monthly_observed = (
        document_scores.groupby("data", as_index=False)
        .agg(
            escore_llm=("escore_llm", "mean"),
            n_documentos_llm=("escore_llm", "count"),
            escore_lexico=("escore_lexico", "mean"),
            n_documentos=("documento_id", "nunique"),
        )
        .sort_values("data")
    )
    month_index = pd.date_range(
        document_scores["data"].min(),
        document_scores["data"].max(),
        freq="MS",
        name="data",
    )
    monthly = (
        monthly_observed.set_index("data")
        .reindex(month_index)
        .rename_axis("data")
        .reset_index()
    )
    monthly[["n_documentos_llm", "n_documentos"]] = (
        monthly[["n_documentos_llm", "n_documentos"]].fillna(0).astype(int)
    )
    monthly.to_csv(
        DATA_DIR / "sentimento_mensal.csv", index=False, encoding="utf-8-sig"
    )
    by_source = (
        document_scores.groupby(["data", "tipo_fonte"], as_index=False)
        .agg(
            escore_llm=("escore_llm", "mean"),
            n_documentos_llm=("escore_llm", "count"),
            escore_lexico=("escore_lexico", "mean"),
            n_documentos=("documento_id", "nunique"),
        )
        .sort_values(["data", "tipo_fonte"])
    )
    by_source.to_csv(
        DATA_DIR / "sentimento_mensal_por_fonte.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dictionary = pd.DataFrame(
        [
            {
                "variavel": "escore_llm / sentimento_llm_lag1",
                "descricao": "Tom de política monetária: negativo expansionista, zero neutro e positivo contracionista.",
                "fonte": "LLM configurado pelo usuário; textos do BCB",
                "unidade": "escore contínuo em [-1, 1]",
                "frequencia": "trecho, documento, mês e painel mensal por área/grupo",
                "tratamento": f"Prompt versionado como {PROMPT_VERSION}; médias mensais usam somente documentos integralmente classificados; o painel usa defasagem de um mês e repete o escore nacional.",
            },
            {
                "variavel": "escore_lexico / sentimento_lexico_lag1",
                "descricao": "Escore de referência pela diferença normalizada entre termos contracionistas e expansionistas.",
                "fonte": "Léxico fixo em src/analise_sentimento.py",
                "unidade": "escore contínuo em [-1, 1]",
                "frequencia": "trecho, documento, mês e painel mensal por área/grupo",
                "tratamento": "Comparador simples; zero também pode significar ausência de termos reconhecidos. No painel, a média nacional é defasada um mês.",
            },
            {
                "variavel": "n_documentos_llm / n_documentos_llm_lag1",
                "descricao": "Número de documentos com classificação LLM completa publicados no mês.",
                "fonte": "API aberta do Banco Central do Brasil",
                "unidade": "documentos",
                "frequencia": "mensal e painel mensal por área/grupo",
                "tratamento": "Ausência de nova publicação é registrada como zero contagem, não como escore neutro; a variável do painel conta documentos publicados no mês anterior.",
            },
        ]
    )
    dictionary.to_csv(
        DATA_DIR / "dicionario_sentimento.csv",
        index=False,
        encoding="utf-8-sig",
    )
    panel_path = DATA_DIR / "painel_ipca_macro.csv"
    if panel_path.exists():
        panel = pd.read_csv(
            panel_path, encoding="utf-8-sig", parse_dates=["data"]
        )
        panel_months = pd.date_range(
            panel["data"].min(), panel["data"].max(), freq="MS"
        )
        time_features = (
            monthly.set_index("data")
            .reindex(panel_months)
            .rename_axis("data")
        )
        time_features["sentimento_llm_lag1"] = (
            time_features["escore_llm"].shift(1).ffill()
        )
        time_features["n_documentos_llm_lag1"] = (
            time_features["n_documentos_llm"].shift(1).fillna(0)
        )
        time_features["sentimento_lexico_lag1"] = (
            time_features["escore_lexico"].shift(1).ffill()
        )
        time_features["n_documentos_lag1"] = (
            time_features["n_documentos"].shift(1).fillna(0)
        )
        panel_features = time_features[
            [
                "sentimento_llm_lag1",
                "n_documentos_llm_lag1",
                "sentimento_lexico_lag1",
                "n_documentos_lag1",
            ]
        ].reset_index()
        panel_with_sentiment = panel.merge(
            panel_features, on="data", how="left", validate="many_to_one"
        )
        panel_with_sentiment.to_csv(
            DATA_DIR / "painel_ipca_macro_sentimento.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(
        f"Agregação concluída: {len(document_scores)} documentos e "
        f"{len(monthly_observed)} meses com publicação."
    )
    return document_scores, monthly


def _cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    compared = pd.DataFrame({"first": first, "second": second}).dropna()
    if compared.empty:
        return float("nan")
    labels = sorted(set(compared["first"]) | set(compared["second"]))
    observed = float((compared["first"] == compared["second"]).mean())
    expected = sum(
        float((compared["first"] == label).mean())
        * float((compared["second"] == label).mean())
        for label in labels
    )
    return (observed - expected) / (1.0 - expected) if expected < 1 else float("nan")


def _normalize_manual_label(series: pd.Series) -> pd.Series:
    """Normaliza rótulos manuais textuais ou numéricos (-1, 0, +1) para as 3 classes válidas."""
    numeric_map = {
        "-1": "expansionista",
        "-1.0": "expansionista",
        "0": "neutro",
        "0.0": "neutro",
        "1": "contracionista",
        "+1": "contracionista",
        "1.0": "contracionista",
        "+1.0": "contracionista",
    }
    cleaned = series.astype("string").fillna("").str.strip().str.lower()
    return cleaned.replace(numeric_map)


def validar_rotulacao() -> pd.DataFrame:
    """Exige classificação LLM de todos os documentos e calcula kappa contra rótulos humanos."""
    manual_path = DATA_DIR / "rotulacao_manual.csv"
    chunks_path = DATA_DIR / "trechos_sentimento.csv"
    documents_path = DATA_DIR / "documentos_copom.csv"
    if not manual_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Execute preparar e preencha rotulacao_manual.csv.")

    chunks = pd.read_csv(chunks_path, encoding="utf-8-sig")
    documents = (
        pd.read_csv(documents_path, encoding="utf-8-sig")
        if documents_path.exists()
        else None
    )
    _assert_all_chunks_classified(chunks, documents=documents)

    manual = pd.read_csv(manual_path, encoding="utf-8-sig")
    manual["rotulo_manual"] = _normalize_manual_label(manual["rotulo_manual"])
    labeled = manual.loc[manual["rotulo_manual"].isin(VALID_LABELS)].copy()
    if len(labeled) < MIN_MANUAL_LABELS:
        raise ValueError(
            f"Foram encontrados {len(labeled)} rótulos manuais válidos; "
            f"são necessários pelo menos {MIN_MANUAL_LABELS}."
        )
    if "documento_id" in labeled and labeled["documento_id"].nunique() < MIN_MANUAL_LABELS:
        raise ValueError(
            "Os rótulos devem cobrir pelo menos 200 documentos distintos."
        )

    labeled = labeled.merge(
        chunks[["trecho_id", "escore_llm", "rotulo_llm", "escore_lexico", "rotulo_lexico"]],
        on="trecho_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for name, column in (
        ("llm", "rotulo_llm"),
        ("lexico_referencia", "rotulo_lexico"),
    ):
        valid = labeled.loc[labeled[column].isin(VALID_LABELS)]
        if len(valid) < MIN_MANUAL_LABELS:
            raise ValueError(
                f"Há apenas {len(valid)} comparações válidas para {name}; "
                f"são necessárias pelo menos {MIN_MANUAL_LABELS}."
            )
        rows.append(
            {
                "comparacao": f"manual_vs_{name}",
                "n_textos": len(valid),
                "kappa_cohen": _cohen_kappa(
                    valid["rotulo_manual"], valid[column]
                ),
            }
        )
    valid_llm_lex = labeled.loc[
        labeled["rotulo_llm"].isin(VALID_LABELS)
        & labeled["rotulo_lexico"].isin(VALID_LABELS)
    ]
    rows.append(
        {
            "comparacao": "llm_vs_lexico_referencia",
            "n_textos": len(valid_llm_lex),
            "kappa_cohen": _cohen_kappa(
                valid_llm_lex["rotulo_llm"], valid_llm_lex["rotulo_lexico"]
            ),
        }
    )
    if "rotulo_manual_2" in labeled:
        second = _normalize_manual_label(labeled["rotulo_manual_2"])
        valid = labeled.loc[second.isin(VALID_LABELS)].copy()
        valid["rotulo_manual_2"] = second.loc[valid.index]
        rows.append(
            {
                "comparacao": "anotador_1_vs_anotador_2",
                "n_textos": len(valid),
                "kappa_cohen": _cohen_kappa(
                    valid["rotulo_manual"], valid["rotulo_manual_2"]
                ),
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(
        DATA_DIR / "validacao_sentimento.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(report.to_string(index=False))
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("coletar", help="Baixa atas e comunicados oficiais.")
    collect.add_argument("--desde", default=DEFAULT_START_DATE)
    commands.add_parser("preparar", help="Cria trechos e amostra de rotulagem manual.")
    classify = commands.add_parser(
        "classificar", help="Classifica todos os trechos pendentes com LLM configurado."
    )
    classify.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Quantidade de trechos enviados por chamada JSON (padrão: 25).",
    )
    classify.add_argument(
        "--intervalo",
        type=float,
        default=4.5,
        help="Intervalo mínimo em segundos entre chamadas à API (padrão: 4.5s).",
    )
    classify.add_argument(
        "--modelo",
        default=None,
        help="Modelo específico para sobrescrever COPOM_LLM_MODEL nesta execução.",
    )
    import_cmd = commands.add_parser(
        "importar-json",
        help="Importa rótulos LLM prontos de um arquivo JSON (ex.: rotulo_llm.json).",
    )
    import_cmd.add_argument(
        "--arquivo",
        default=str(DATA_DIR / "rotulo_llm.json"),
        help="Caminho do arquivo JSON com trecho_id, escore, confianca e justificativa.",
    )
    import_cmd.add_argument(
        "--modelo",
        default="llm_externo_json",
        help="Identificador do modelo a registrar em modelo_llm.",
    )
    commands.add_parser("agregar", help="Gera escores por documento e mês (exige classificação completa).")
    commands.add_parser("validar", help="Exige corpus classificado e calcula kappa com rotulação manual.")
    args = parser.parse_args()

    if args.command == "coletar":
        coletar_documentos(args.desde)
    elif args.command == "preparar":
        preparar_trechos()
    elif args.command == "classificar":
        classificar_trechos(
            batch_size=args.batch_size,
            min_interval_seconds=args.intervalo,
            model_override=args.modelo,
        )
    elif args.command == "importar-json":
        importar_rotulos_json(Path(args.arquivo), model_name=args.modelo)
    elif args.command == "agregar":
        agregar_sentimento()
    elif args.command == "validar":
        validar_rotulacao()


if __name__ == "__main__":
    _main()
