# Rodando no Kaggle

Passo a passo para executar `notebooks/fact_extractor_distill.ipynb` numa GPU grátis do Kaggle.

## Antes de começar (uma vez só)

1. **Suba este repositório para o GitHub** como `DevTheusP/fact-extractor-distill` (público). O
   notebook clona o código de lá; se usar outro nome, ajuste `REPO_URL` na célula de configuração.
   O agent-memory-bench já está público em `DevTheusP/agent-memory-bench`.
2. **Verifique o telefone na conta do Kaggle** (Settings > Phone verification). Sem isso o Kaggle não
   libera GPU nem internet no notebook.
3. **(Opcional, só para publicar no Hugging Face)** crie um token de escrita em
   huggingface.co > Settings > Access Tokens. Você mesmo cria e cola; o notebook nunca pede o token
   em texto.

## Criar o notebook

1. kaggle.com > **Create > New Notebook**.
2. **File > Import Notebook** e escolha `notebooks/fact_extractor_distill.ipynb` (do seu clone, ou
   pela URL do arquivo no GitHub).
3. No painel lateral, **Settings**:
   - **Accelerator: GPU T4 x2** (a P100 também funciona, mas a T4 x2 deixa o embedder do benchmark
     numa GPU separada do vLLM);
   - **Internet: On** (precisa para `pip`, GitHub e Hugging Face);
   - **Persistence**: pode deixar desligada; os artefatos vão para `/kaggle/working/outputs`.
4. **(Opcional)** **Add-ons > Secrets > Add a new secret**: nome `HF_TOKEN`, valor = seu token.
   Marque o secret como anexado a este notebook. Depois mude `PUSH_TO_HUB = True` e confira
   `HF_REPO_MODEL` e `HF_REPO_DATASET` na célula de configuração.

## Rodar

1. Primeiro com `SMOKE_TEST = True` e **Run All**. Roda o pipeline inteiro com 20 exemplos e 10 passos
   de treino; o tempo vai quase todo em instalação, download dos modelos e subida do vLLM.
2. Se terminar sem erro, mude para `SMOKE_TEST = False`, **Restart & Run All**.
3. Para deixar público e com saídas: **Save Version > Save & Run All (Commit)**. A versão roda em
   segundo plano (pode fechar a aba) e vira o notebook público do portfólio.
4. Copie os números de `results/RESULTS.md` (também em `/kaggle/working/outputs/results/`) para as
   tabelas `TBD` do README.

## Tempo estimado (GPU T4 x2)

| etapa | smoke test | completo (0,5B) |
|---|---|---|
| instalação (pip + venv do vLLM) | 8–12 min | 8–12 min |
| geração dos dados | < 1 min | < 1 min |
| professor: download (~5,5 GB) + subida do vLLM | 5–8 min | 5–8 min |
| professor: rotulagem (~3.600 pedidos) | 1 min | 10–15 min |
| professor: avaliação + benchmark | 3 min | 10–15 min |
| aluno base: avaliação + benchmark | 4 min | 8–10 min |
| treino LoRA + merge | 2 min | 15–25 min |
| aluno destilado: avaliação + benchmark | 4 min | 8–10 min |
| GGUF (compilar llama-quantize + converter) | 8–10 min | 8–10 min |
| **total** | **~35–45 min** | **~1h30–2h** |

Com `ALSO_STUDENT_1_5B = True`, some mais ~1 h (avaliação do base, treino de ~40 min, avaliação do
destilado). Tudo cabe numa sessão (limite de 12 h) e gasta umas 2–3 h da cota semanal de 30 h.

## Se algo der errado

**O vLLM não sobe na T4.** O log completo fica em `/kaggle/working/logs/vllm_<modelo>.log`.
- erro de memória ou de CUDA graphs: `VLLM_ENFORCE_EAGER = True`;
- erro de kernel/backend de atenção em Turing (sm_75): tente a versão anterior do vLLM em
  `VLLM_VERSION` (apague `/kaggle/tmp/vllm-env` antes: `!rm -rf /kaggle/tmp/vllm-env`);
- se nenhuma versão subir, o plano B é o `HFEngine` de `distill/engines.py` (transformers puro, sem
  gramática): `HFEngine("Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True)` para o professor, chamado
  com `distill.label.label_examples` direto numa célula. Fica bem mais lento e a saída não é guiada.

**Falta de espaço.** Modelos, venvs e cache do Hugging Face ficam em `/kaggle/tmp`; só os artefatos
finais vão para `/kaggle/working` (limite de 20 GB).

**A sessão caiu no meio.** As etapas gravam em disco (`data/labeled`, `results/`, modelos em
`/kaggle/tmp/models`); rode de novo só as células que faltam. A rotulagem é a etapa mais cara: se
`data/labeled/sft_train.jsonl` existir, pule a célula do professor.

**O push para o Hub falhou.** Confira se o secret se chama exatamente `HF_TOKEN`, se está anexado ao
notebook e se o token tem permissão de escrita. Os artefatos continuam em `/kaggle/working/outputs`.
