"""
Pipeline de ingestão: download → parse → chunk → embed → store.

Uso:
    python -m ingestion.pipeline --months 3
    python -m ingestion.pipeline --start 2024-01-01 --end 2024-03-31
    python -m ingestion.pipeline --date 2024-05-01   # ingestão de um dia (cron diário)
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()

console = Console()


def run_pipeline(start: date, end: date, sections: list[str] | None = None) -> None:
    from indexing.chunker import chunk_ato
    from indexing.embedder import embed_documents
    from indexing.store import get_connection, init_schema, upsert_ato, upsert_chunks
    from ingestion.dou_client import INLABSClient
    from ingestion.parser import parse_xml

    if sections is None:
        sections = ["DO1", "DO2", "DO3"]

    conn = get_connection()
    init_schema(conn)

    with INLABSClient() as client:
        current = start
        while current <= end:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            for sec in sections:
                console.print(f"[cyan]Baixando DOU {current.isoformat()} seção {sec}[/cyan]")
                xml_bytes = client.download_xml(current, sec)
                if not xml_bytes:
                    console.print(f"  [dim]Sem publicação[/dim]")
                    continue

                from ingestion.parser import parse_xml
                atos = parse_xml(xml_bytes, current, sec)
                console.print(f"  {len(atos)} atos encontrados")

                for ato in track(atos, description=f"  Indexando seção {sec}..."):
                    upsert_ato(conn, ato)

                    chunks = chunk_ato(ato)
                    if not chunks:
                        continue

                    embeddings = embed_documents(chunks)
                    upsert_chunks(
                        conn,
                        ato_id=ato.id,
                        chunks=chunks,
                        embeddings=embeddings,
                        meta={
                            "orgao": ato.orgao,
                            "secao": ato.secao,
                            "data": ato.data,
                            "tipo_ato": ato.tipo_ato,
                            "fonte": ato.fonte,
                        },
                    )

            current += timedelta(days=1)

    conn.close()
    console.print("[green]Pipeline concluído.[/green]")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingestão do DOU")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--months", type=int, default=3, help="Últimos N meses")
    group.add_argument("--start", type=date.fromisoformat)
    group.add_argument("--date", type=date.fromisoformat, help="Ingestão de um único dia")
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--sections", nargs="+", default=["DO1", "DO2", "DO3"])
    args = parser.parse_args()

    if args.date:
        run_pipeline(args.date, args.date, args.sections)
    elif args.start:
        end = args.end or date.today()
        run_pipeline(args.start, end, args.sections)
    else:
        from ingestion.dou_client import last_n_months
        start, end = last_n_months(args.months)
        run_pipeline(start, end, args.sections)


if __name__ == "__main__":
    main()
