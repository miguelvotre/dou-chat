"""
Cliente para download de XMLs do DOU via INLABS (inlabs.in.gov.br).

Fluxo:
1. Autenticar com email/senha para obter JWT
2. Baixar XML por data e seção
3. Parsear atos do XML

Cadastro gratuito em: https://inlabs.in.gov.br/
Documentação: https://inlabs.in.gov.br/doc
"""

import os
import zipfile
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

import re

BASE_URL = "https://inlabs.in.gov.br"
SECTIONS = ["DO1", "DO2", "DO3", "DO1E", "DO2E", "DO3E"]  # fallback estático


class INLABSClient:
    def __init__(self):
        self.email = os.environ["INLABS_EMAIL"]
        self.password = os.environ["INLABS_PASSWORD"]
        self._token: str | None = None
        self._client = httpx.Client(timeout=60)

    def authenticate(self) -> None:
        resp = self._client.post(
            f"{BASE_URL}/logar.php",
            data={"email": self.email, "password": self.password},
            headers={"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        session_cookie = self._client.cookies.get("inlabs_session_cookie")
        if not session_cookie:
            raise RuntimeError("Falha na autenticação INLABS: cookie de sessão não encontrado")
        self._token = session_cookie

    def _ensure_auth(self) -> None:
        if not self._token:
            self.authenticate()

    def list_available_sections(self, pub_date: date) -> list[str]:
        """
        Descobre dinamicamente quais seções/edições estão disponíveis para a data.
        Faz parsing da página de índice do INLABS e retorna lista de seções, ex:
          ["DO1", "DO2", "DO3"] ou ["DO1", "DO2", "DO3", "DO1E"]
        """
        self._ensure_auth()
        date_str = pub_date.strftime("%Y-%m-%d")
        page = self._client.get(
            f"{BASE_URL}/index.php?p={date_str}", follow_redirects=True
        )
        page.raise_for_status()
        # extrai nomes dos ZIPs: ex. "2026-05-07-DO1.zip" → "DO1"
        zips = re.findall(
            rf"{date_str}-([A-Z0-9]+)\.zip", page.text
        )
        # preserva ordem de aparição, remove duplicatas
        seen: set[str] = set()
        result = []
        for s in zips:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result or SECTIONS  # fallback se parsing falhar

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def download_xml(self, pub_date: date, section: str) -> bytes | None:
        """
        Baixa o ZIP com XML do DOU para uma data e seção.
        Retorna os bytes do XML ou None se não houver publicação.
        """
        self._ensure_auth()
        date_str = pub_date.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/index.php?p={date_str}&dl={date_str}-{section}.zip"

        resp = self._client.get(url, follow_redirects=True)

        if resp.status_code == 404:
            return None  # sem publicação nessa data/seção
        resp.raise_for_status()

        # INLABS retorna um ZIP com um XML por ato — combina todos num único documento
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
            if not xml_files:
                return None
            articles = []
            for fname in xml_files:
                content = zf.read(fname).decode("utf-8", errors="replace")
                # cada arquivo é <xml><article ...>...</article></xml> — extrai só o <article>
                start = content.find("<article")
                end = content.rfind("</article>")
                if start != -1 and end != -1:
                    articles.append(content[start:end + len("</article>")])
            if not articles:
                return None
            combined = "<xml>" + "".join(articles) + "</xml>"
            return combined.encode("utf-8")

    def download_range(
        self,
        start: date,
        end: date,
        sections: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Baixa XMLs para um intervalo de datas e lista de seções.
        Salva em output_dir e retorna lista de paths.
        """
        if sections is None:
            sections = ["DO1", "DO2", "DO3"]
        if output_dir is None:
            output_dir = Path("data/raw")

        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []

        current = start
        while current <= end:
            # DOU não é publicado aos fins de semana
            if current.weekday() < 5:
                for sec in sections:
                    xml_bytes = self.download_xml(current, sec)
                    if xml_bytes:
                        fname = output_dir / f"dou_{current.isoformat()}_secao{sec}.xml"
                        fname.write_bytes(xml_bytes)
                        saved.append(fname)
            current += timedelta(days=1)

        return saved

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def last_n_months(n: int = 3) -> tuple[date, date]:
    today = date.today()
    start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    for _ in range(n - 1):
        start = (start - timedelta(days=1)).replace(day=1)
    return start, today
