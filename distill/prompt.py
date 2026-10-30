"""Prompt e schema do extrator, copiados sem alteração do agent-memory-bench.

Fonte: https://github.com/DevTheusP/agent-memory-bench, arquivo memory/extract.py
(licença MIT, mesmo autor). O professor e o aluno recebem exatamente o mesmo
SYSTEM, o mesmo SCHEMA e a mesma mensagem de usuário que o LLMExtractor monta;
qualquer diferença aqui invalidaria a comparação com o benchmark. O teste
tests/test_prompt.py confere a cópia byte a byte quando o repositório original
está disponível ao lado deste.
"""

from __future__ import annotations

SYSTEM = """Você extrai fatos duráveis de mensagens enviadas a um assistente de código que \
acompanha um projeto de software.

Extraia apenas fatos que continuarão valendo nas próximas semanas: configuração do projeto, \
decisões do time, convenções, ferramentas usadas e regras. Ignore relatos do dia a dia, \
opiniões passageiras e qualquer coisa que a própria mensagem diga não se aplicar.

Para cada fato:
- subject_key: chave estável no formato area.atributo, em inglês e snake_case \
(exemplo: billing.database_engine). Se uma das chaves existentes descreve o mesmo atributo, \
reutilize-a exatamente. Partes diferentes do projeto têm chaves diferentes.
- value: o valor atual, curto.
- kind: "procedural" para regras sobre como agir ou o que nunca fazer; "semantic" para os demais.
- statement: uma frase autocontida, em português, que descreve o fato.

Se não houver fato durável, devolva uma lista vazia."""

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject_key": {"type": "string"},
                    "value": {"type": "string"},
                    "kind": {"type": "string", "enum": ["semantic", "procedural"]},
                    "statement": {"type": "string"},
                },
                "required": ["subject_key", "value", "kind", "statement"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def build_user(text: str, existing_keys: list[str]) -> str:
    """Mesma montagem de LLMExtractor.extract no benchmark."""
    keys = "\n".join(f"- {k}" for k in existing_keys) or "(nenhuma ainda)"
    return f"Chaves existentes:\n{keys}\n\nMensagem:\n{text}"


def build_messages(text: str, existing_keys: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user(text, existing_keys)},
    ]
