# fact-extractor-distill

**Um modelo 14x menor consegue fazer o papel de extrator de memória de um agente?**

No [agent-memory-bench](https://github.com/DevTheusP/agent-memory-bench), a política de memória
`v2_keyed` usa um LLM para transformar cada mensagem do usuário em fatos duráveis com uma chave de
sujeito (`subject_key`). A chave é o que resolve atualização na escrita: quando o serviço migra de
região, o fato novo substitui o antigo em vez de competir com ele na busca. Com o Qwen2.5-7B como
extrator, a recuperação média foi de **98%**, contra **62%** da busca ingênua por similaridade.

Só que um 7B para cada mensagem é caro. Este projeto destila esse extrator para o
**Qwen2.5-0.5B-Instruct** (e, opcionalmente, para o 1.5B) com fine-tuning LoRA e mede duas coisas:

1. **fidelidade**: o aluno devolve os mesmos fatos que o professor (JSON válido, F1 de fatos,
   reutilização de chave, `kind`, lista vazia quando não há fato)?
2. **efeito no sistema**: trocando o professor pelo aluno, a recuperação do benchmark se mantém?

Tudo o que precisa de GPU roda num notebook público do Kaggle (T4); localmente ficam o código, os
testes e a geração de dados.

## Resultados

Rodada de 2026-10-30, a segunda no Kaggle (T4 x2): professor Qwen2.5-7B-Instruct-AWQ no vLLM,
aluno Qwen2.5-0.5B-Instruct com LoRA r=16, 2 épocas, treinado com os 2433 rótulos de treino que
passaram nos verificadores (de 3000). Os arquivos da rodada estão em
[`results/kaggle-run2/`](results/kaggle-run2/), e o modelo e o dataset estão no Hugging Face:
[modelo](https://huggingface.co/DevTheusP/qwen2.5-0.5b-fact-extractor-ptbr) ·
[dataset](https://huggingface.co/datasets/DevTheusP/fact-extractor-distill-ptbr).

A primeira rodada (sem verificadores) deu 82–83% de recuperação para todos os extratores. O que
estava errado e como foi corrigido está em [`docs/diagnostico.md`](docs/diagnostico.md).

### Extração: teste sintético com áreas e regras fora do treino, referência = gabarito do template

| modelo | parâmetros | schema válido (guiado) | schema válido (sem gramática) | F1 por valor | F1 estrito | kind | lista vazia certa | latência p50 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (base) | 0,5B | 99,3% | 0% | 36,1% | 4,9% | 36,9% | 6,9% | 0,69 s |
| **Qwen2.5-0.5B destilado** | 0,5B | **100%** | **99,7%** | **66,1%** | **25,7%** | **87,2%** | **97,2%** | **0,28 s** |
| Qwen2.5-7B-Instruct-AWQ (professor) | 7B | 100% | 0% | 61,9% | 23,5% | 62,1% | 94,4% | 1,50 s |

- O aluno passa o professor no gabarito porque aprendeu dos rótulos já filtrados: sem chave
  existente sem relação, sem statement em inglês e com o `kind` corrigido.
- "Sem gramática", o professor devolve uma lista (`[...]`) em vez de `{"facts": [...]}`: o prompt
  do benchmark não mostra esse formato, só a gramática garante. O aluno aprendeu o formato no treino.
- F1 estrito exige a mesma `subject_key`; os nomes de chave variam muito entre modelos, por isso o
  F1 por valor é a métrica principal.

### agent-memory-bench, camada 1 (recuperação média, grade publicada: 5 seeds × 5 ruídos × 5 pausas)

| extrator | v1_naive | v2_keyed | v5_guarded | regra rara (v2 → v5) | falhas de JSON |
|---|---|---|---|---|---|
| publicado: qwen2.5:7b no Ollama | 62,0% | 98,0% | 99,6%* | 92,0% → 98,3%* | 0 |
| professor (7B AWQ, vLLM) | 62,0% | 83,4% | 97,9% | 33,7% → 91,4% | 0 |
| aluno base 0,5B | 62,0% | 83,3% | 97,6% | 33,1% → 91,4% | 58 |
| **aluno destilado 0,5B** | 62,0% | **100%** | **100%** | **100% → 100%** | 0 |

\* reprocessado em CPU com o cache de extração publicado (ver o diagnóstico).

Como ler:

- A `v1_naive` não usa extrator e dá o mesmo número em todas as linhas: é o controle.
- Na v2, o professor perde a regra rara porque escreveu o statement em inglês, e a similaridade
  com a pergunta (em português) cai abaixo de um ruído que se repete toda semana. O aluno escreveu
  em português e ficou acima, **mas por pouco**: 0,649 contra 0,635. O resultado robusto é o da v5.
- Com a v5, até o aluno base chega a 97,6%. A camada de recuperação mede mais a política de memória
  do que a qualidade do extrator; para comparar extratores, valem as métricas de extração acima.

## Método

### 1. Dados: templates com seed fixa, não geração pelo professor

`distill/generate.py` gera as mensagens por combinatória de templates em PT-BR (seed 13): 22 atributos
de projeto (banco, região, linguagem, fila, CI, SLO...) em 22 áreas, 23 regras ("nunca faça X",
mínimo de aprovações, cobertura), 12 convenções, ruído do dia a dia, opiniões passageiras, mensagens que negam a própria aplicabilidade
("isso era de outro projeto"), conteúdo externo (README, changelog, saída de ferramenta, inclusive
com instruções para "AI assistants"), mensagens com 2 ou 3 fatos e listas de chaves existentes que
devem ser reutilizadas (inclusive com nomes alternativos, como `orders_service.database`) ou que são
distratores (o mesmo atributo de outra área).

Por que templates e não pedir ao próprio professor para escrever as mensagens:

- **a intenção é conhecida**: cada exemplo sabe quais fatos afirma. Isso dá uma segunda referência,
  independente do professor, para medir a qualidade dos rótulos e para avaliar o próprio professor;
- **controle de vazamento**: dá para provar que nada dos cenários do benchmark entrou no treino.
  Um professor gerando texto livremente tende a reproduzir justamente os exemplos canônicos
  (região da AWS, framework de teste, force push);
- **reprodutível e barato**: os mesmos 3 mil exemplos saem em 2 segundos, em qualquer máquina, sem GPU;
- **estratificável**: as métricas saem por categoria (ruído, regra, reutilização...).

O custo é a diversidade linguística menor que a de texto livre. O teste sintético mitiga parte disso
usando 4 áreas, 4 regras e 2 convenções que não aparecem no treino, e o teste do benchmark (texto
escrito à mão, de outra fonte) mede a generalização de verdade.

| conjunto | tamanho | origem |
|---|---|---|
| treino | 3.000 | templates, áreas e regras comuns |
| validação | 200 | templates, sem texto em comum com o treino |
| teste sintético | 300 | templates com áreas, regras e convenções reservadas |
| teste do benchmark | 131 | pares (mensagem, chaves) do cache da rodada publicada |

### 2. Vazamento

As mensagens dos cenários do agent-memory-bench são o teste, nunca o treino. Elas estão copiadas,
com atribuição, em `distill/resources/bench/` (licença MIT, mesmo autor). O gerador:

- não tem a área de pagamentos nem os pares `backend`/`frontend` × framework de teste;
- não fala de force push nem de verificação TLS (`BANNED_PATTERNS` em `distill/leakage.py`);
- descarta qualquer mensagem com similaridade de Jaccard (tokens) acima de 0,35 com uma mensagem do
  benchmark, e as frases de abertura dos cenários ("Regra do time, importante:", "Terminamos a
  migração") também estão banidas.

`tests/test_generate.py` confere tudo isso a cada execução.

### 3. Rotulagem pelo professor

O professor é o **Qwen2.5-7B-Instruct-AWQ** servido pelo **vLLM 0.32** (venv separado, `--dtype half`
porque a T4 não tem bfloat16). Ele recebe exatamente o `SYSTEM`, o `SCHEMA` e a mensagem de usuário do
`LLMExtractor` do benchmark (`distill/prompt.py`, copiado de `memory/extract.py`; um teste compara
byte a byte com o original). A saída é guiada pelo schema (`response_format: json_schema` na API
compatível com OpenAI, que no vLLM atual substituiu o antigo `guided_json`), com temperatura 0.
Rótulos fora do schema são descartados do treino.

Por que AWQ: o 7B em fp16 ocupa ~15 GB e não cabe numa T4 de 16 GB com cache de KV; o AWQ 4-bit ocupa
~5,5 GB e o vLLM tem kernel AWQ para Turing (sm_75). A rodada publicada também usou um 7B quantizado
(q4 do Ollama), então os dois professores são comparáveis, mas não idênticos: o teste do benchmark
mede também a concordância entre eles (`ref=professor_ollama`).

### 4. Formatação e treino

Formato conversacional prompt-completion do TRL: `prompt = [system, user]`, `completion = [assistant]`,
com o JSON do professor em uma linha. A perda fica só na resposta (`completion_only_loss`), o que
dispensa as tags `{% generation %}` que `assistant_only_loss` exigiria no template do Qwen2.5.
O notebook confere que `distill/formatting.py` (ChatML em Python puro, usado nos testes) gera o mesmo
texto que `tokenizer.apply_chat_template`.

**TRL + PEFT, não Unsloth.** O 0,5B cabe com folga numa T4 em LoRA comum (treino de 10 a 20 minutos),
então o ganho de velocidade do Unsloth não compensa o risco de ele remendar versões do transformers
no ambiente do Kaggle. TRL e PEFT são a referência da Hugging Face e o adaptador sai num formato que
qualquer um carrega.

| hiperparâmetro | valor |
|---|---|
| LoRA | r=16, alpha=32, dropout 0,05, em q/k/v/o/gate/up/down |
| épocas | 2 |
| learning rate | 2e-4, cosseno, warmup de 5% |
| batch efetivo | 16 (8 × 2 de acumulação) |
| precisão | base em float32, autocast fp16 (a T4 não tem bf16) |
| comprimento máximo | 1024 tokens (o maior exemplo tem ~730) |

### 5. Métricas (`distill/metrics.py`)

| métrica | definição |
|---|---|
| JSON válido / schema válido | a resposta crua é JSON / segue o `SCHEMA` do benchmark. Fora do schema conta como lista vazia nas demais |
| F1 estrito | fato certo = `subject_key` normalizada igual **e** `value` normalizado igual, pareamento um para um |
| F1 por valor | só o `value` (igual ou um contido no outro): o fato certo com outro nome de chave |
| reutilização de chave | quando a referência reutiliza uma chave existente, o modelo usa exatamente a mesma string? O store do benchmark busca a chave exata |
| reutilização indevida | o modelo usa uma chave existente que a referência não usou (ex.: a chave de banco de outra área) |
| kind | acerto de `semantic`/`procedural` nos pares casados por valor |
| lista vazia certa | quando a referência é vazia (ruído, negação, conteúdo externo), o modelo também devolve vazia? |
| latência p50/p95, tokens/s | um pedido por vez numa amostra de 50, como o benchmark chama o extrator |

Cada modelo é medido contra três referências: o **professor** (fidelidade da destilação), o
**gabarito** do template ou do cenário (vale também para o professor) e, no teste do benchmark, os
rótulos da **rodada publicada** no Ollama. E em dois modos: `guided` (com gramática, como o benchmark
usa) e `free` (sem gramática).

### 6. Benchmark com o aluno no lugar do professor

`distill/adapter.py` implementa a interface `LLM` do benchmark (`complete_json(system, user, schema)`)
sobre o modelo servido no vLLM, e troca o embedder `bge-m3` do Ollama pelo mesmo `BAAI/bge-m3` via
sentence-transformers. `distill/bench.py` importa o runner do benchmark clonado e roda a camada 1
sem modificar uma linha dele.

## Como reproduzir

### No Kaggle (GPU)

Siga o [KAGGLE.md](KAGGLE.md): importar o notebook, ligar GPU T4 e internet, opcionalmente adicionar
o secret `HF_TOKEN`, e rodar tudo. Comece com `SMOKE_TEST = True` (~35–45 min, quase tudo é instalação
e download) e depois rode completo (~2 h com o 0,5B; ~3 h com o 1.5B também).

### Localmente (CPU: código, dados e testes)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                     # nenhum teste precisa de GPU ou internet
.venv/bin/python -m distill.generate --out data/generated
```

Se o [agent-memory-bench](https://github.com/DevTheusP/agent-memory-bench) estiver clonado ao lado
deste repositório (ou em `BENCH_REPO`), os testes também conferem o prompt contra o original e rodam a
camada 1 do benchmark com um extrator falso.

Para usar o modelo destilado no Ollama, depois de baixar o GGUF do Hugging Face (pasta `gguf/`):

```bash
ollama create fact-extractor-0.5b -f Modelfile
uv run python -m eval.run --llm fact-extractor-0.5b   # dentro do agent-memory-bench
```

### Estrutura

```
distill/
  prompt.py       SYSTEM e SCHEMA copiados do benchmark
  generate.py     gerador de mensagens sintéticas (templates, seed fixa)
  leakage.py      checagem de vazamento contra o benchmark
  bench_data.py   conjunto de teste do benchmark (mensagens + cache publicado)
  label.py        rotulagem pelo professor e dataset de SFT
  formatting.py   formato prompt-completion e ChatML do Qwen
  parsing.py      leitura e validação da saída do modelo
  metrics.py      métricas de extração
  engines.py      cliente da API compatível com OpenAI (vLLM) e plano B com transformers
  server.py       sobe e derruba o vllm serve
  evaluate.py     avaliação de um modelo servido
  adapter.py      interface LLM/Embedder do agent-memory-bench
  bench.py        camada 1 do benchmark com qualquer extrator
  train.py        LoRA com TRL + PEFT
  export.py       merge, GGUF q4, Modelfile e push para o Hub
  report.py       tabelas e gráficos
notebooks/        notebook do Kaggle (gerado por build_notebook.py)
hf/               model card e dataset card do Hugging Face
tests/            testes locais (pytest)
```

## Limitações

- **Diversidade dos dados.** Templates cobrem bem a estrutura do problema, mas o vocabulário é
  limitado. O aluno pode ir bem no teste sintético e pior em mensagens reais escritas de outro jeito;
  o teste do benchmark é a melhor medida disso, e ele é pequeno.
- **Teste do benchmark pequeno e desbalanceado.** 131 pares, só 35 textos distintos e a maioria ruído:
  os fatos de verdade são poucos (as 4 mensagens dos cenários aparecem algumas vezes com listas de
  chaves diferentes). Um erro muda vários pontos percentuais; os intervalos de confiança são largos.
- **O alvo é o professor, com os erros dele.** Destilação por rótulo copia também os enganos do 7B.
  Os verificadores (`label_issues`, `repair_kinds`) tiram os três vícios medidos, mas só esses: um
  erro do professor que não viole nenhum verificador continua indo para o aluno.
- **Limiar e deduplicação da v5.** O limiar de 0,55 foi calibrado em 3 substituições, e a
  deduplicação só junta textos idênticos; ruído parafraseado continuaria competindo pelas vagas.
- **Gabarito com chaves canônicas.** O nome da chave é arbitrário (o professor escolhe), então contra
  o gabarito só o F1 por valor, a reutilização e a lista vazia são informativos; o F1 estrito contra o
  gabarito não é.
- **Professor do Kaggle ≠ professor publicado.** AWQ no vLLM e q4 no Ollama não dão saídas idênticas,
  e o bge-m3 via sentence-transformers não é bit a bit o do Ollama.
- **Latência medida em T4 com vLLM.** Os números servem para comparar os modelos entre si, não
  para prever latência em CPU ou no Ollama.
- **Só a camada 1.** A camada 2 do benchmark (resposta do agente) depende de o modelo de resposta
  obedecer à memória, o que não muda com o extrator; ficou fora.

## Licença

Código sob [MIT](LICENSE). O modelo destilado herda a Apache-2.0 do Qwen2.5-0.5B-Instruct, e o dataset
é rotulado pelo Qwen2.5-7B-Instruct (Apache-2.0). As mensagens do benchmark em
`distill/resources/bench/` vêm do agent-memory-bench (MIT).
