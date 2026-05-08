"""
TruLens evaluation queries.

Represent typical lawyer searches in the DOU.
Expand with real queries after user validation.
"""

# (query, optional filters)
EVAL_QUERIES: list[tuple[str, dict]] = [
    # Legal topic search
    ("Quais são as portarias recentes sobre contratação de pessoal no setor público?", {"tipo_ato": "portaria"}),
    ("Resolução sobre prazo para recursos administrativos", {"tipo_ato": "resolucao"}),
    ("Instruções normativas sobre obrigações tributárias para pessoa jurídica", {"tipo_ato": "instrucao_normativa"}),

    # Search by issuing body
    ("Atos recentes do Ministério da Fazenda sobre política fiscal", {"orgao": "Ministério da Fazenda"}),
    ("Editais do Ministério da Saúde para compras e licitações", {"orgao": "Ministério da Saúde", "tipo_ato": "edital"}),

    # Search by entity / person
    ("Nomeações e exonerações em cargos de confiança do Poder Executivo", {"secao": "2"}),
    ("Contratos firmados com empresas de tecnologia da informação", {}),

    # Date-scoped search
    ("Decretos presidenciais publicados na seção 1", {"secao": "1", "tipo_ato": "decreto"}),
    ("Extratos de contratos do Ministério da Defesa", {"orgao": "Ministério da Defesa", "tipo_ato": "extrato"}),

    # Complex semantic search
    ("regulamentação de proteção de dados pessoais no serviço público", {}),
    ("penalidades para servidores públicos por improbidade administrativa", {}),
    ("licitação dispensada por valor abaixo do limite legal", {}),
]
