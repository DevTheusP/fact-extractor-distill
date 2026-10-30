---
license: apache-2.0
language:
- pt
pretty_name: Fact extractor distill (PT-BR)
size_categories:
- 1K<n<10K
task_categories:
- text-generation
tags:
- distillation
- synthetic
- information-extraction
- structured-output
- agent-memory
configs:
- config_name: sft
  data_files:
  - split: train
    path: data/sft_train.jsonl
  - split: validation
    path: data/sft_val.jsonl
- config_name: labeled
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/val.jsonl
  - split: test_synth
    path: data/test_synth.jsonl
  - split: test_bench
    path: data/test_bench.jsonl
---

# Fact extractor distill (PT-BR)

Mensagens sintéticas em português do Brasil enviadas a um assistente de código, rotuladas pelo
**Qwen2.5-7B-Instruct-AWQ** com os fatos duráveis que cada uma afirma. Serve para destilar o extrator
de fatos da política `v2_keyed` do [agent-memory-bench](https://github.com/DevTheusP/agent-memory-bench)
para modelos pequenos. Código e notebook: [fact-extractor-distill](https://github.com/DevTheusP/fact-extractor-distill).

## Configurações

**`sft`**: pronto para o `SFTTrainer` do TRL, no formato conversacional prompt-completion:

```json
{"id": "train-3f2a...", "prompt": [{"role": "system", "content": "Você extrai fatos duráveis..."},
 {"role": "user", "content": "Chaves existentes:\n- orders.database_engine\n\nMensagem:\n..."}],
 "completion": [{"role": "assistant", "content": "{\"facts\": [...]}"}]}
```

**`labeled`**: o exemplo completo, com metadados e o rótulo bruto do professor:

| campo | descrição |
|---|---|
| `text` | a mensagem |
| `existing_keys` | chaves de memória que já existem (o modelo deve reutilizar a que descreve o mesmo atributo) |
| `category` | `config`, `update`, `key_reuse`, `reinforcement`, `rule`, `convention`, `noise`, `opinion`, `negated`, `external`, `multi`, `mixed_negation` (ou `bench_*` no teste do benchmark) |
| `intents` | o que o template afirma (chave canônica, valor, kind); referência independente do professor |
| `reuse_key` / `distractor_keys` | chave existente que deve ser reutilizada / que não deve |
| `teacher` | saída crua do professor, se é JSON e schema válidos, fatos, latência e tokens |

## Divisões

| divisão | exemplos | observação |
|---|---|---|
| train | 3.000 | áreas e regras comuns |
| val | 200 | sem texto em comum com o treino |
| test_synth | 300 | áreas, regras e convenções que não aparecem no treino |
| test_bench | 131 | pares (mensagem, chaves) do agent-memory-bench; nunca usar no treino |

## Como foi feito

1. **Geração:** templates com combinatória e seed fixa (`distill/generate.py`). A escolha por
   templates, e não por geração livre pelo LLM, garante intenção conhecida, reprodutibilidade e
   controle de vazamento.
2. **Vazamento:** as áreas e atributos dos cenários do benchmark estão fora do gerador, e mensagens
   com similaridade de Jaccard acima de 0,35 com qualquer mensagem do benchmark são descartadas.
3. **Rotulagem:** Qwen2.5-7B-Instruct-AWQ no vLLM, com o mesmo prompt e o mesmo JSON Schema do
   benchmark, saída guiada e temperatura 0. Rótulos fora do schema ficam fora da configuração `sft`.

Qualidade dos rótulos (professor contra a intenção do template), medida no notebook:

<!-- RESULTADOS -->

## Limitações e vieses

- Texto de templates: vocabulário e estrutura de frase limitados.
- Os rótulos são de um LLM e carregam os erros dele; a coluna `intents` permite filtrar ou auditar.
- Domínio estreito: projetos de software, português do Brasil.
- O `test_bench` vem de outro projeto (MIT) e está aqui só para avaliação.

## Licença

Apache-2.0. Os rótulos foram gerados pelo Qwen2.5-7B-Instruct (Apache-2.0). As mensagens de
`test_bench` vêm do agent-memory-bench (MIT, mesmo autor).
