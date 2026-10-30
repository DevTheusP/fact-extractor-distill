# Diagnóstico: por que a primeira rodada no Kaggle deu 83% e não 98%

Na primeira execução completa do notebook (2026-10-30), a camada 1 do agent-memory-bench
deu 83,4% de recuperação média na `v2_keyed` com o professor, 82,3% com o aluno destilado
e 83,3% com o aluno sem treino. A rodada publicada do benchmark (qwen2.5:7b no Ollama)
tinha dado 98%. Todos os experimentos abaixo rodaram em CPU, na grade publicada
(5 seeds × 5 níveis de ruído × 5 pausas).

## 1. O embedder e o adaptador não são a causa

| experimento | v2_keyed | regra rara |
|---|---|---|
| cache de extração publicado + bge-m3 via sentence-transformers | 98,0% | 92,0% |
| as mesmas respostas passando pelo `EngineLLM` (adaptador do Kaggle) | 98,0% | 92,0% |
| respostas do professor do Kaggle (AWQ/vLLM) passando pelo mesmo caminho | 83,4% | 33,7% |
| aluno destilado ao vivo (transformers, CPU) | 82,3% | 29,1% |

A `v1_naive`, que não usa extrator, dá 62,0% em todas as linhas, igual à rodada publicada.
A diferença está no que o extrator escreve.

## 2. Primeiro mecanismo: um fato inventado desliga a regra

Trace do aluno destilado. A regra vira `coding_rules.force_push = never` (correto). Semanas
depois, a mensagem *"Terminamos a migração: a partir de hoje o serviço de pagamentos roda em
sa-east-1"* gera a região certa e mais um fato que não existe na mensagem:

```
coding_rules.force_push = false  ("The force push rule is disabled for this project.")
```

A chave já existe e o valor é novo, então a `v2_keyed` trata como atualização e desativa a
regra. Na v2, esse fato falso aparece no contexto em **275 de 875** perguntas sobre a regra.

O vício vem do professor: nos 3000 rótulos de treino, **217 (7%)** escrevem valor numa chave
existente que não tem relação com a mensagem. Também há 469 (16%) com `statement` em inglês,
contra o prompt, e 285 (10%) com `kind` diferente do template.

## 3. Segundo mecanismo (o principal): a regra perde para um ruído repetido

Similaridade (bge-m3) entre a pergunta da sonda (*"Fiz rebase da main local e o push foi
rejeitado..."*) e cada candidato:

| texto | similaridade |
|---|---|
| "A never force push na main deve ser seguida." (statement da rodada publicada) | **0,657** |
| "Fiz rebase da minha branch de feature em cima da main." (ruído semanal) | 0,635 |
| qualquer outra redação da regra, em PT ou EN | 0,537 a 0,622 |

O ruído se repete toda semana; com ruído alto, ocupa as 5 vagas sozinho. Na rodada publicada,
a regra só vencia porque o 7B escreveu um statement quebrado que, por acaso, ficou mais
parecido com a pergunta. **Os 92% da regra rara publicados dependiam de sorte de redação.**

## 4. Correção na memória: `v5_guarded` (agent-memory-bench)

- **Escrita protegida**: um fato só substitui outro se a mensagem de origem for parecida com
  o statement antigo (cosseno ≥ 0,55). Calibração: substituição inventada 0,39; legítimas
  (região) 0,72 e 0,75. São só 3 casos, então o limiar é provisório.
- **Leitura sem repetição**: memórias com o mesmo texto ocupam uma vaga só.

| extrator | v2 | v5 | só deduplicação | só proteção |
|---|---|---|---|---|
| qwen2.5:7b publicado | 98,0% | **99,6%** | 99,6% | 98,0% |
| aluno destilado 0.5B | 82,3% | **97,9%** | 97,9% | 83,4% |

Na métrica de recuperação, quem resolve é a deduplicação. A proteção não muda o número porque
o episódio original da regra continua guardado e conta como acerto, mas tira o fato falso do
contexto (275 → 0 de 875). A métrica não pune fato contraditório no contexto; isso é uma
limitação do benchmark, não uma prova de que a proteção é inútil.

## 5. Correção nos dados (este repositório)

- `label_issues` rejeita rótulos com chave existente sem relação e com statement em inglês;
  `repair_kinds` corrige o `kind` pelo gabarito do template. Nos rótulos da primeira rodada,
  sobram 2408 de 3000.
- Nova categoria `rule_guard`: mudança ou configuração de uma área com regras na lista de
  chaves existentes, que não podem ser tocadas.
- `N_TRAIN` sobe de 3000 para 4000 para compensar os descartes.
- O notebook passa a publicar no dataset as predições, os relatórios e os caches de extração
  do benchmark: a sessão do Kaggle apaga `/kaggle/working`, e a primeira rodada perdeu tudo isso.

## 6. Segunda rodada no Kaggle, com os verificadores

Os verificadores tiraram 242 rótulos com chave sem relação e 325 com statement em inglês, e
corrigiram o `kind` de 298 (sobraram 2433 de 3000 no treino). O aluno re-treinado não inventou
nenhum fato no benchmark e escreveu tudo em português:

| extrator | v2_keyed | v5_guarded |
|---|---|---|
| professor (7B AWQ) | 83,4% | 97,9% |
| aluno base | 83,3% | 97,6% |
| aluno destilado com verificadores | 100% | 100% |

O 100% na v2 tem margem pequena: o statement do aluno ("Nunca realizar forcer push na main.")
fica em 0,673 contra 0,635 do ruído, e 0,649 sem o erro de digitação. O professor escreveu a
regra em inglês (0,537) e perdeu. A comparação robusta entre extratores são as métricas de
extração no README.

## Limitações

- O limiar 0,55 foi calibrado em 3 substituições do próprio benchmark.
- A deduplicação é por texto idêntico; ruído parafraseado continuaria competindo.
- O replay do professor da primeira rodada usou cache incompleto (68 de 131 chamadas sem
  resposta gravada); a seção 6 mede o professor ao vivo.
