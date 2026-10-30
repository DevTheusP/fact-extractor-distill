# Resultados

## test_bench|ref=gabarito · guided

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_bench|ref=gabarito | guided | 100.0% | 0.0% | 6.3% | 100.0% | 14.3% | 4.1% | 0.0% | 0.72 | 159.16 |
| aluno-destilado | test_bench|ref=gabarito | guided | 100.0% | 10.0% | 30.0% | 100.0% | 66.7% | 95.9% | 50.0% | 0.06 | 118.62 |
| professor | test_bench|ref=gabarito | guided | 100.0% | 10.5% | 42.1% | 100.0% | 100.0% | 97.5% | 40.0% | 0.26 | 35.69 |

## test_bench|ref=gabarito · free

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_bench|ref=gabarito | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.26 | 153.83 |
| aluno-destilado | test_bench|ref=gabarito | free | 100.0% | 10.0% | 30.0% | 100.0% | 66.7% | 95.9% | 50.0% | 0.05 | 127.10 |
| professor | test_bench|ref=gabarito | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.08 | 34.00 |

## test_bench|ref=professor · guided

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_bench|ref=professor | guided | 100.0% | 1.7% | 7.5% | 100.0% | 11.1% | 0.0% | 0.0% | 0.72 | 159.16 |
| aluno-destilado | test_bench|ref=professor | guided | 100.0% | 28.6% | 76.2% | 75.0% | 75.0% | 98.3% | 27.3% | 0.06 | 118.62 |
| professor | test_bench|ref=professor | guided | 100.0% | 90.9% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.26 | 35.69 |

## test_bench|ref=professor · free

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_bench|ref=professor | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.26 | 153.83 |
| aluno-destilado | test_bench|ref=professor | free | 100.0% | 28.6% | 76.2% | 75.0% | 75.0% | 98.3% | 27.3% | 0.05 | 127.10 |
| professor | test_bench|ref=professor | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.08 | 34.00 |

## test_bench|ref=professor_ollama · guided

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_bench|ref=professor_ollama | guided | 100.0% | 1.7% | 5.9% | 100.0% | 28.6% | 0.0% | 0.0% | 0.72 | 159.16 |
| aluno-destilado | test_bench|ref=professor_ollama | guided | 100.0% | 31.6% | 63.2% | 100.0% | 83.3% | 96.7% | 33.3% | 0.06 | 118.62 |
| professor | test_bench|ref=professor_ollama | guided | 100.0% | 50.0% | 60.0% | 100.0% | 100.0% | 96.7% | 22.2% | 0.26 | 35.69 |

## test_bench|ref=professor_ollama · free

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_bench|ref=professor_ollama | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.26 | 153.83 |
| aluno-destilado | test_bench|ref=professor_ollama | free | 100.0% | 31.6% | 63.2% | 100.0% | 83.3% | 96.7% | 33.3% | 0.05 | 127.10 |
| professor | test_bench|ref=professor_ollama | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.08 | 34.00 |

## test_synth|ref=gabarito · guided

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_synth|ref=gabarito | guided | 99.3% | 4.9% | 36.1% | 58.2% | 36.9% | 6.9% | 0.0% | 0.69 | 154.43 |
| aluno-destilado | test_synth|ref=gabarito | guided | 100.0% | 25.7% | 66.1% | 70.9% | 87.2% | 97.2% | 26.3% | 0.28 | 144.76 |
| professor | test_synth|ref=gabarito | guided | 100.0% | 23.5% | 61.9% | 76.4% | 62.1% | 94.4% | 30.7% | 1.50 | 37.52 |

## test_synth|ref=gabarito · free

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_synth|ref=gabarito | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.30 | 157.37 |
| aluno-destilado | test_synth|ref=gabarito | free | 99.7% | 25.7% | 65.2% | 70.9% | 87.1% | 97.2% | 26.3% | 0.28 | 150.97 |
| professor | test_synth|ref=gabarito | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 1.30 | 37.52 |

## test_synth|ref=professor · guided

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_synth|ref=professor | guided | 99.3% | 4.6% | 30.2% | 57.8% | 42.3% | 0.0% | 0.0% | 0.69 | 154.43 |
| aluno-destilado | test_synth|ref=professor | guided | 100.0% | 25.4% | 62.4% | 53.1% | 61.8% | 80.0% | 13.3% | 0.28 | 144.76 |
| professor | test_synth|ref=professor | guided | 100.0% | 97.7% | 99.2% | 100.0% | 99.5% | 100.0% | 0.6% | 1.50 | 37.52 |

## test_synth|ref=professor · free

| modelo | conjunto | modo | schema_valid | strict_f1 | value_f1 | reuse_accuracy | kind_accuracy | empty_accuracy | false_empty_rate | latency_p50_s | tokens_per_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aluno-base | test_synth|ref=professor | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 0.30 | 157.37 |
| aluno-destilado | test_synth|ref=professor | free | 99.7% | 25.9% | 61.9% | 53.1% | 61.5% | 80.0% | 13.3% | 0.28 | 150.97 |
| professor | test_synth|ref=professor | free | 0.0% | - | - | 0.0% | - | 0.0% | 0.0% | 1.30 | 37.52 |

## agent-memory-bench, camada 1 (recuperação)

| extrator | memória | retrieval | ret_update | ret_scope | ret_rare_fact | ret_injection | falhas de JSON |
|---|---|---|---|---|---|---|---|
| aluno-base | v1_naive | 62.0% | 100.0% | 100.0% | 33.7% | 0.0% | 58 |
| aluno-base | v2_keyed | 83.3% | 100.0% | 100.0% | 33.1% | 100.0% | 58 |
| aluno-base | v3_decay_rank | 53.4% | 99.4% | 12.8% | 1.5% | 100.0% | 58 |
| aluno-base | v4_decay_tier | 53.1% | 100.0% | 8.6% | 3.8% | 100.0% | 58 |
| aluno-base | v5_guarded | 97.6% | 99.0% | 100.0% | 91.4% | 100.0% | 58 |
| aluno-destilado | v1_naive | 62.0% | 100.0% | 100.0% | 33.7% | 0.0% | 0 |
| aluno-destilado | v2_keyed | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0 |
| aluno-destilado | v3_decay_rank | 71.7% | 96.4% | 54.4% | 36.6% | 100.0% | 0 |
| aluno-destilado | v4_decay_tier | 69.4% | 100.0% | 37.5% | 40.0% | 100.0% | 0 |
| aluno-destilado | v5_guarded | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0 |
| professor | v1_naive | 62.0% | 100.0% | 100.0% | 33.7% | 0.0% | 0 |
| professor | v2_keyed | 83.4% | 100.0% | 100.0% | 33.7% | 100.0% | 0 |
| professor | v3_decay_rank | 54.7% | 97.5% | 14.1% | 7.4% | 100.0% | 0 |
| professor | v4_decay_tier | 55.2% | 100.0% | 8.6% | 12.1% | 100.0% | 0 |
| professor | v5_guarded | 97.9% | 100.0% | 100.0% | 91.4% | 100.0% | 0 |
| publicado (Ollama, qwen2.5:7b) | v2_keyed | 98% | | | | | |
| publicado (Ollama, qwen2.5:7b) | v1_naive | 62% | | | | | |
