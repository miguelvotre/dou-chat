"""
Parser dos XMLs do DOU (formato INLABS real).

Estrutura real do XML INLABS:
  <xml>
    <article id="..." artCategory="Ministério..." pubName="DO1" artType="..." pubDate="DD/MM/YYYY" ...>
      <body>
        <Identifica><![CDATA[...]]></Identifica>
        <Titulo><![CDATA[...]]></Titulo>
        <SubTitulo><![CDATA[...]]></SubTitulo>
        <Texto><![CDATA[<p>...</p>]]></Texto>
        <Assina><![CDATA[...]]></Assina>
        <Cargo><![CDATA[...]]></Cargo>
      </body>
    </article>
  </xml>
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from lxml import etree


@dataclass
class Ato:
    """Representa um ato publicado no DOU."""

    id: str
    identificador: str
    titulo: str
    subtitulo: str
    texto: str
    orgao: str
    secao: str
    data: date
    tipo_ato: str
    assina: str
    cargo: str
    fonte: str = "federal"
    uf: str | None = None
    municipio: str | None = None
    url_original: str | None = None

    @property
    def texto_completo(self) -> str:
        parts = [self.titulo, self.subtitulo, self.texto]
        return "\n".join(p for p in parts if p)


def _strip_html(html: str) -> str:
    """Remove tags HTML e normaliza espaços."""
    text = re.sub(r"<[^>]+>", " ", html or "")
    return " ".join(text.split())


def _cdata(el) -> str:
    if el is None:
        return ""
    return _strip_html(el.text or "")


def _infer_tipo_ato(identifica: str, art_type: str) -> str:
    texto = (identifica + " " + art_type).lower()
    mapa = {
        "portaria": "portaria",
        "resolução": "resolucao",
        "resolucao": "resolucao",
        "instrução normativa": "instrucao_normativa",
        "instrucao normativa": "instrucao_normativa",
        "despacho": "despacho",
        "edital": "edital",
        "aviso": "aviso",
        "extrato": "extrato",
        "contrato": "contrato",
        "decreto": "decreto",
        "lei ": "lei",
        "medida provisória": "medida_provisoria",
        "nomeação": "nomeacao",
        "nomeacao": "nomeacao",
        "exoneração": "exoneracao",
        "pauta": "pauta",
    }
    for termo, tipo in mapa.items():
        if termo in texto:
            return tipo
    return "outros"


def parse_xml(xml_bytes: bytes, pub_date: date, section: str) -> list[Ato]:
    """Parseia um XML do INLABS e retorna lista de Atos."""
    root = etree.fromstring(xml_bytes)
    atos: list[Ato] = []

    for article in root.iter("article"):
        art_id = article.get("id", "")
        art_category = article.get("artCategory", "").strip()
        art_type = article.get("artType", "").strip()
        pdf_page = article.get("pdfPage", "")

        body = article.find("body")
        if body is None:
            continue

        identifica = _cdata(body.find("Identifica"))
        titulo = _cdata(body.find("Titulo"))
        subtitulo = _cdata(body.find("SubTitulo"))
        texto = _cdata(body.find("Texto"))
        assina = _cdata(body.find("Assina"))
        cargo = _cdata(body.find("Cargo"))

        if not texto and not identifica:
            continue

        ato = Ato(
            id=f"{pub_date.isoformat()}_{section}_{art_id}",
            identificador=art_id,
            titulo=identifica or titulo,
            subtitulo=subtitulo,
            texto=texto,
            orgao=art_category,
            secao=section,
            data=pub_date,
            tipo_ato=_infer_tipo_ato(identifica, art_type),
            assina=assina,
            cargo=cargo,
            url_original=pdf_page or None,
        )
        atos.append(ato)

    return atos


def parse_file(path: Path) -> list[Ato]:
    """Parseia um arquivo XML salvo em disco."""
    stem = path.stem  # dou_2024-01-15_DO1
    parts = stem.split("_")
    pub_date = date.fromisoformat(parts[1])
    section = parts[2]
    return parse_xml(path.read_bytes(), pub_date, section)
