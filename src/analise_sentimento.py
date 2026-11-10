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


def _llm_configuration() -> tuple[str, str, str]:
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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = session.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Resposta do LLM sem conteúdo textual.")
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    result = json.loads(content)
    score = float(result["escore"])
    confidence = float(result["confianca"])
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError(f"Escore LLM fora de [-1, 1]: {score}")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(f"Confiança LLM fora de [0, 1]: {confidence}")
    return score, confidence, str(result["justificativa"]).strip(), _label_for_score(score)


def classificar_trechos(sample_only: bool = False) -> pd.DataFrame:
    """Classifica apenas trechos ainda sem escore usando API OpenAI-compatible."""
    base_url, api_key, model = _llm_configuration()
    chunks = preparar_trechos()
    session = requests.Session()
    is_pending = pd.to_numeric(chunks["escore_llm"], errors="coerce").isna()
    if sample_only:
        manual_path = DATA_DIR / "rotulacao_manual.csv"
        manual = pd.read_csv(manual_path, encoding="utf-8-sig")
        selected_ids = set(manual["trecho_id"])
        is_pending &= chunks["trecho_id"].isin(selected_ids)
    pending = chunks.index[is_pending]
    for count, index in enumerate(pending, start=1):
        result = _classify_with_llm(
            session,
            base_url,
            api_key,
            model,
            str(chunks.at[index, "trecho_texto"]),
        )
        chunks.loc[
            index,
            [
                "escore_llm",
                "confianca_llm",
                "justificativa_llm",
                "rotulo_llm",
                "modelo_llm",
                "versao_prompt",
            ],
        ] = [
            *result[:3],
            result[3],
            model,
            PROMPT_VERSION,
        ]
        if count % 10 == 0 or count == len(pending):
            chunks.to_csv(
                DATA_DIR / "trechos_sentimento.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(f"Classificados {count}/{len(pending)} trechos pendentes.")

    chunks.to_csv(
        DATA_DIR / "trechos_sentimento.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Classificação concluída com o modelo {model}.")
    return chunks


def agregar_sentimento() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega escores por documento e mês de publicação."""
    chunks_path = DATA_DIR / "trechos_sentimento.csv"
    documents_path = DATA_DIR / "documentos_copom.csv"
    if not chunks_path.exists() or not documents_path.exists():
        raise FileNotFoundError("Execute os comandos coletar e preparar primeiro.")
    chunks = pd.read_csv(chunks_path, encoding="utf-8-sig")
    documents = pd.read_csv(
        documents_path, encoding="utf-8-sig", parse_dates=["data_publicacao"]
    )
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


def validar_rotulacao() -> pd.DataFrame:
    """Calcula kappa contra o LLM e o léxico sem fabricar rótulos humanos."""
    manual_path = DATA_DIR / "rotulacao_manual.csv"
    chunks_path = DATA_DIR / "trechos_sentimento.csv"
    if not manual_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Execute preparar e preencha rotulacao_manual.csv.")
    manual = pd.read_csv(manual_path, encoding="utf-8-sig")
    manual["rotulo_manual"] = (
        manual["rotulo_manual"].astype("string").fillna("").str.strip().str.lower()
    )
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

    chunks = pd.read_csv(chunks_path, encoding="utf-8-sig")
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
    if "rotulo_manual_2" in labeled:
        second = (
            labeled["rotulo_manual_2"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )
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
        "classificar", help="Classifica trechos com LLM configurado."
    )
    classify.add_argument(
        "--somente-amostra",
        action="store_true",
        help="Classifica apenas os 200 trechos separados para validação humana.",
    )
    commands.add_parser("agregar", help="Gera escores por documento e mês.")
    commands.add_parser("validar", help="Calcula kappa após a rotulagem manual.")
    args = parser.parse_args()

    if args.command == "coletar":
        coletar_documentos(args.desde)
    elif args.command == "preparar":
        preparar_trechos()
    elif args.command == "classificar":
        classificar_trechos(sample_only=args.somente_amostra)
    elif args.command == "agregar":
        agregar_sentimento()
    elif args.command == "validar":
        validar_rotulacao()


if __name__ == "__main__":
    _main()
