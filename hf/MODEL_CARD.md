---
license: apache-2.0
language:
- pt
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
datasets:
- DevTheusP/fact-extractor-distill-ptbr
tags:
- distillation
- lora
- peft
- trl
- information-extraction
- structured-output
- json
- agent-memory
- gguf
---

# Qwen2.5-0.5B fact extractor (PT-BR)

Extrator de fatos duráveis para a memória de um agente de código, destilado do
**Qwen2.5-7B-Instruct** (professor) para o **Qwen2.5-0.5B-Instruct** (aluno) com LoRA.

Ele recebe uma mensagem enviada a um assistente de código e a lista de chaves de memória que já
existem, e devolve um JSON com os fatos que continuam valendo nas próximas semanas (configuração,
decisões, convenções, regras), cada um com uma `subject_key` estável no formato `area.atributo`.
Quando uma chave existente descreve o mesmo atributo, ele a reutiliza: é isso que permite à memória
substituir um fato antigo em vez de guardar os dois.

Feito para a política `v2_keyed` do [agent-memory-bench](https://github.com/DevTheusP/agent-memory-bench).
Código, dados e notebook: [fact-extractor-distill](https://github.com/DevTheusP/fact-extractor-distill).

## Conteúdo do repositório

| caminho | o que é |
|---|---|
| raiz | modelo merged (fp16, safetensors), pronto para `transformers` ou vLLM |
| `adapter/` | só o adaptador LoRA, para aplicar sobre o `Qwen/Qwen2.5-0.5B-Instruct` |
| `gguf/` | GGUF q4_K_M e `Modelfile` para Ollama ou llama.cpp |
| `train_log.json` | hiperparâmetros e curva de perda |

## Como usar

O modelo foi treinado com um prompt fixo; use exatamente o `SYSTEM` abaixo (o mesmo do benchmark)
e a mensagem de usuário no formato `Chaves existentes:` / `Mensagem:`.

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "DevTheusP/qwen2.5-0.5b-fact-extractor-ptbr"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)

from huggingface_hub import hf_hub_download
SYSTEM = open(hf_hub_download(repo, "system_prompt.txt")).read()  # o mesmo prompt do benchmark
user = "Chaves existentes:\n- billing.database_engine\n\nMensagem:\nDecidimos trocar o banco do faturamento: sai MySQL, entra PostgreSQL."
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=256, do_sample=False)
print(json.loads(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
# {"facts": [{"subject_key": "billing.database_engine", "value": "PostgreSQL", "kind": "semantic", ...}]}
```

Com Ollama (pasta `gguf/`): `ollama create fact-extractor-0.5b -f Modelfile`. Para saída sempre
válida, passe o JSON Schema no campo `format` (Ollama) ou `response_format` (vLLM), como o benchmark faz.

## Treino

- **Dados:** 3.000 mensagens sintéticas em PT-BR geradas por templates com seed fixa, rotuladas pelo
  Qwen2.5-7B-Instruct-AWQ (vLLM, saída guiada pelo schema, temperatura 0). Nenhuma mensagem do
  benchmark entra no treino.
- **Método:** destilação por rótulo (SFT na resposta do professor), LoRA r=16, alpha=32, em todas as
  projeções lineares; 2 épocas, lr 2e-4 cosseno, batch efetivo 16, fp16, perda só na resposta.
- **Bibliotecas:** TRL 1.17, PEFT 0.21, transformers 5.22.
- **Hardware:** 1× NVIDIA T4 (Kaggle).

## Resultados

Medidos no notebook do Kaggle. Referência = rótulos do professor, salvo indicação.

<!-- RESULTADOS -->

## Limitações

- Treinado só com mensagens sintéticas de templates: o vocabulário é limitado e o desempenho em
  mensagens reais escritas de outro jeito pode ser menor.
- Copia também os erros do professor (destilação por rótulo).
- O prompt é fixo; com outro `SYSTEM` ou outro formato de mensagem, o comportamento não foi avaliado.
- Só português do Brasil e o domínio de projetos de software.
- Não use como única defesa contra injeção de prompt: no benchmark, conteúdo externo de baixa
  confiança nem chega ao extrator (quarentena).

## Licença

Apache-2.0, a mesma do modelo base Qwen2.5-0.5B-Instruct. Os rótulos de treino vêm do
Qwen2.5-7B-Instruct, também Apache-2.0.
