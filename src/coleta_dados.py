"""Coleta e estrutura dados públicos de inflação e variáveis macroeconômicas."""

from __future__ import annotations

import calendar
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dados" / "tratados"
IBGE_API = "https://servicodados.ibge.gov.br/api/v3/agregados"
BCB_SGS_API = "https://api.bcb.gov.br/dados/serie/bcdata.sgs"
FOCUS_API = (
    "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
    "ExpectativaMercadoMensais"
)

GRUPOS_IPCA = {
    "7170": "Alimentação e bebidas",
    "7445": "Habitação",
    "7486": "Artigos de residência",
    "7558": "Vestuário",
    "7625": "Transportes",
    "7660": "Saúde e cuidados pessoais",
    "7712": "Despesas pessoais",
    "7766": "Educação",
    "7786": "Comunicação",
}

SERIES_BCB = {
    "selic_efetiva_aa": {
        "codigo": "4189",
        "nome": "Taxa Selic efetiva acumulada no mês, anualizada base 252",
        "unidade": "% a.a.",
        "agregacao": "taxa efetiva mensal anualizada, conforme série publicada pelo BCB",
    },
    "cambio_usd_brl": {
        "codigo": "3697",
        "nome": "Taxa de câmbio comercial de venda (dólar)",
        "unidade": "R$/US$",
        "agregacao": "média mensal publicada pelo BCB",
    },
    "ic_br": {
        "codigo": "27574",
        "nome": "Índice de Commodities - Brasil (IC-Br)",
        "unidade": "índice",
        "agregacao": "valor mensal publicado pelo BCB",
    },
    "ibc_br_sa": {
        "codigo": "24364",
        "nome": "IBC-Br dessazonalizado",
        "unidade": "índice",
        "agregacao": "valor mensal publicado pelo BCB",
    },
    "desocupacao": {
        "codigo": "24369",
        "nome": "Taxa de desocupação (PNAD Contínua)",
        "unidade": "%",
        "agregacao": "valor mensal publicado pelo BCB",
    },
    "ipca_brasil": {
        "codigo": "433",
        "nome": "IPCA - variação mensal (Brasil)",
        "unidade": "%",
        "agregacao": "valor mensal publicado pelo BCB",
    },
}


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "projeto-ipca-copom/1.0"})
    return session


def _get_json(session: requests.Session, url: str, **kwargs: Any) -> Any:
    response = session.get(url, timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def _period_from_key(key: str) -> pd.Timestamp:
    if not re.fullmatch(r"\d{6}", key):
        raise ValueError(f"Período SIDRA inesperado: {key!r}")
    return pd.to_datetime(key + "01", format="%Y%m%d")


def _number_or_missing(value: Any) -> float | None:
    text = str(value).strip()
    if text in {"", "-", "...", "…"}:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Valor numérico inesperado na API: {value!r}") from exc


def _metadata_period(
    session: requests.Session, table_id: int
) -> tuple[str, str]:
    metadata = _get_json(session, f"{IBGE_API}/{table_id}/metadados")
    periodicity = metadata["periodicidade"]
    return str(periodicity["inicio"]), str(periodicity["fim"])


def _collect_ipca(session: requests.Session) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    table_parts = ((1419, "201201"), (7060, "202001"))

    for table_id, minimum_period in table_parts:
        table_start, table_end = _metadata_period(session, table_id)
        period_start = max(minimum_period, table_start)
        if int(table_end) < int(period_start):
            continue

        for level in ("N6", "N7"):
            params = {
                "localidades": f"{level}[all]",
                "classificacao": f"315[{','.join(GRUPOS_IPCA)}]",
            }
            url = (
                f"{IBGE_API}/{table_id}/periodos/"
                f"{period_start}-{table_end}/variaveis/63"
            )
            payload = _get_json(session, url, params=params)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError(
                    f"Resposta SIDRA inesperada para tabela {table_id}, "
                    f"nível {level}: {type(payload).__name__}"
                )
            result = payload[0]
            received_groups: set[str] = set()

            for group_data in result["resultados"]:
                category = group_data["classificacoes"][0]["categoria"]
                if len(category) != 1:
                    raise ValueError(
                        f"Categoria SIDRA inesperada na tabela {table_id}: "
                        f"{category}"
                    )
                group_id, group_name = next(iter(category.items()))
                if group_id not in GRUPOS_IPCA:
                    continue
                received_groups.add(group_id)

                for series in group_data["series"]:
                    locality = series["localidade"]
                    for period_key, value in series["serie"].items():
                        rows.append(
                            {
                                "data": _period_from_key(period_key),
                                "area_id": str(locality["id"]),
                                "area": locality["nome"],
                                "nivel_geografico": level,
                                "grupo_id": group_id,
                                "grupo_despesa": re.sub(
                                    r"^\d+\.", "", group_name
                                ),
                                "variacao_ipca_mensal_pct": _number_or_missing(
                                    value
                                ),
                                "tabela_sidra": table_id,
                            }
                        )

            missing_groups = set(GRUPOS_IPCA) - received_groups
            if missing_groups:
                raise ValueError(
                    f"Tabela SIDRA {table_id}, nível {level}: grupos ausentes "
                    f"{sorted(missing_groups)}"
                )
            print(
                f"SIDRA {table_id}, {level}: {len(result['resultados'])} grupos "
                f"consultados; {len(rows):,} observações acumuladas."
            )

    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("A API do SIDRA não retornou dados de IPCA.")
    if data.duplicated(["data", "area_id", "grupo_id"]).any():
        raise ValueError("A coleta SIDRA produziu chaves duplicadas.")
    return data.sort_values(
        ["data", "nivel_geografico", "area", "grupo_id"]
    ).reset_index(drop=True)


def _parse_sgs_date(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, format="%d/%m/%Y")


def _collect_macro(
    session: requests.Session, first_month: pd.Timestamp, last_month: pd.Timestamp
) -> pd.DataFrame:
    first_date = first_month.strftime("%d/%m/%Y")
    last_date = (
        last_month + pd.offsets.MonthEnd(0)
    ).strftime("%d/%m/%Y")
    monthly_index = pd.date_range(first_month, last_month, freq="MS")
    macro = pd.DataFrame({"data": monthly_index})

    for variable, metadata in SERIES_BCB.items():
        url = f"{BCB_SGS_API}.{metadata['codigo']}/dados"
        observations = _get_json(
            session,
            url,
            params={
                "formato": "json",
                "dataInicial": first_date,
                "dataFinal": last_date,
            },
        )
        values = pd.DataFrame(observations)
        if values.empty:
            raise ValueError(
                f"O BCB não retornou dados para a série {metadata['codigo']} "
                f"({variable})."
            )
        values["data"] = values["data"].map(_parse_sgs_date)
        values[variable] = values["valor"].map(_number_or_missing)
        values["data"] = values["data"].dt.to_period("M").dt.to_timestamp()
        monthly = values.groupby("data", as_index=False)[variable].mean()
        macro = macro.merge(monthly, on="data", how="left", validate="one_to_one")
        print(
            f"BCB SGS {metadata['codigo']} ({variable}): "
            f"{values['data'].nunique()} meses com dados."
        )

    return macro


def _last_snapshot_for_month(
    session: requests.Session, year: int, month: int
) -> list[dict[str, Any]]:
    month_end = date(year, month, calendar.monthrange(year, month)[1])
    next_month = (
        date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    )
    window_start = month_end - timedelta(days=10)
    filter_text = (
        "Indicador eq 'IPCA' "
        f"and Data ge '{window_start.isoformat()}' "
        f"and Data lt '{next_month.isoformat()}'"
    )
    encoded_filter = quote(filter_text, safe="'")
    url = (
        f"{FOCUS_API}?%24filter={encoded_filter}"
        "&%24top=10000&%24format=json"
    )
    observations: list[dict[str, Any]] = []

    while url:
        result = _get_json(session, url)
        observations.extend(result.get("value", []))
        url = result.get("@odata.nextLink", "")

    if not observations:
        return []
    latest_date = max(row["Data"] for row in observations)
    return [row for row in observations if row["Data"] == latest_date]


def _collect_focus(
    session: requests.Session, first_month: pd.Timestamp, last_month: pd.Timestamp
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in pd.date_range(first_month, last_month, freq="MS"):
        snapshot = _last_snapshot_for_month(
            session, int(period.year), int(period.month)
        )
        for row in snapshot:
            reference = datetime.strptime(
                row["DataReferencia"], "%m/%Y"
            ).date()
            rows.append(
                {
                    "data_coleta": pd.Timestamp(row["Data"]),
                    "mes_referencia": pd.Timestamp(reference).replace(day=1),
                    "indicador": row["Indicador"],
                    "mediana_pct": row["Mediana"],
                    "media_pct": row["Media"],
                    "desvio_padrao_pct": row["DesvioPadrao"],
                    "respondentes": row["numeroRespondentes"],
                    "base_calculo": row["baseCalculo"],
                }
            )
        if not snapshot:
            print(
                f"AVISO: sem expectativa Focus para {period:%Y-%m} "
                "na janela dos últimos 10 dias do mês."
            )
        time.sleep(0.03)

    focus = pd.DataFrame(rows)
    if not focus.empty and focus.duplicated(
        ["data_coleta", "mes_referencia", "base_calculo"]
    ).any():
        raise ValueError("A coleta Focus produziu chaves duplicadas.")
    return focus.sort_values(
        ["data_coleta", "mes_referencia", "base_calculo"]
    ).reset_index(drop=True)


def _write_dictionary() -> None:
    records = [
        {
            "variavel": "variacao_ipca_mensal_pct",
            "descricao": "Variação mensal do IPCA por área geográfica e grupo de despesa.",
            "fonte": "IBGE/SIDRA",
            "identificador": "Tabelas 1419 (2012-2019) e 7060 (2020 em diante), variável 63",
            "unidade": "%",
            "frequencia": "mensal",
            "periodo": "2012-01 em diante; fim conforme atualização do SIDRA",
            "tratamento": "Valores ausentes do SIDRA permanecem nulos; sem imputação.",
        },
        {
            "variavel": "area_id, area, nivel_geografico",
            "descricao": "Identificador, nome e nível geográfico da área de coleta do IPCA.",
            "fonte": "IBGE/SIDRA",
            "identificador": "N6 e N7 nas tabelas 1419 e 7060",
            "unidade": "não se aplica",
            "frequencia": "mensal",
            "periodo": "áreas disponibilizadas pela tabela",
            "tratamento": "Nível e identificação preservados conforme resposta oficial.",
        },
        {
            "variavel": "grupo_id, grupo_despesa",
            "descricao": "Nove grupos principais de despesas do IPCA.",
            "fonte": "IBGE/SIDRA",
            "identificador": "Classificação 315; categorias 7170, 7445, 7486, 7558, 7625, 7660, 7712, 7766 e 7786",
            "unidade": "categoria",
            "frequencia": "mensal",
            "periodo": "2012-01 em diante",
            "tratamento": "Nomes de categoria mantidos em português.",
        },
    ]
    for variable, metadata in SERIES_BCB.items():
        records.append(
            {
                "variavel": variable,
                "descricao": metadata["nome"],
                "fonte": "Banco Central do Brasil (SGS)",
                "identificador": f"Série SGS {metadata['codigo']}",
                "unidade": metadata["unidade"],
                "frequencia": "mensal",
                "periodo": "2012-01 em diante; pode haver início/fim distinto por série",
                "tratamento": metadata["agregacao"],
            }
        )
    records.append(
        {
            "variavel": "mediana_pct (e demais campos Focus)",
            "descricao": "Expectativas de mercado para IPCA mensal, preservadas por data de coleta, mês de referência e base de cálculo.",
            "fonte": "Banco Central do Brasil (Sistema Expectativas de Mercado/Focus)",
            "identificador": "ExpectativaMercadoMensais, indicador IPCA",
            "unidade": "%",
            "frequencia": "instantâneo mensal (última data disponível em cada mês)",
            "periodo": "2012-01 em diante; conforme histórico da API",
            "tratamento": "Base de cálculo mantida; não juntar ao alvo sem respeitar a data de divulgação.",
        }
    )
    pd.DataFrame(records).to_csv(
        DATA_DIR / "dicionario_variaveis.csv",
        index=False,
        encoding="utf-8-sig",
    )


def coletar_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Baixa as fontes, valida as chaves e grava as bases normalizadas."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()
    ipca = _collect_ipca(session)
    first_month = ipca["data"].min()
    last_month = ipca["data"].max()
    macro = _collect_macro(session, first_month, last_month)
    focus = _collect_focus(session, first_month, last_month)

    ipca.to_csv(DATA_DIR / "ipca_painel.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(DATA_DIR / "macro_mensal.csv", index=False, encoding="utf-8-sig")
    focus.to_csv(
        DATA_DIR / "focus_expectativas.csv", index=False, encoding="utf-8-sig"
    )
    panel = ipca.merge(macro, on="data", how="left", validate="many_to_one")
    panel.to_csv(
        DATA_DIR / "painel_ipca_macro.csv", index=False, encoding="utf-8-sig"
    )
    _write_dictionary()

    print(
        f"Base IPCA: {len(ipca):,} linhas, {ipca['area_id'].nunique()} áreas, "
        f"{ipca['grupo_id'].nunique()} grupos, de {first_month:%Y-%m} "
        f"a {last_month:%Y-%m}."
    )
    print(
        f"Macroeconomia: {len(macro):,} meses; Focus: {len(focus):,} "
        "observações de expectativas."
    )
    return ipca, macro, focus


if __name__ == "__main__":
    coletar_base()
