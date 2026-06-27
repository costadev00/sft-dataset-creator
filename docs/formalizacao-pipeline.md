# Formalizacao da pipeline de criacao e filtragem

Este documento resume o que foi implementado e executado ate aqui no
`sft-dataset-creator`, com foco na pipeline de filtragem das geracoes. A ideia
e servir como base metodologica para a escrita de um artigo: primeiro uma
descricao conceitual do fluxo, depois uma versao mais operacional, com os
artefatos, regras e numeros observados na run `wiki-brazil-all-tasks-22596`.

## Resumo em uma frase

O sistema constroi um dataset sintetico de SFT a partir de documentos da
Wikipedia em portugues, planejando exemplos de forma deterministica, gerando
candidatos estruturados com um unico LLM local e filtrando cada candidato por
regras automaticas verificaveis de formato, auto-contencao, nao duplicacao e
ancoragem em evidencias extraidas do proprio documento.

## Objetivo do projeto

O objetivo pratico foi criar uma pipeline reprodutivel para gerar exemplos de
SFT em portugues brasileiro a partir do corpus `costadev00/wiki-brazil`. Cada
exemplo final deve conter uma instrucao, uma possivel entrada auxiliar, uma
resposta e metadados de proveniencia, incluindo o documento de origem, tipo de
tarefa, dificuldade e spans de evidencia.

O projeto foi desenhado para evitar que o dataset final dependa de julgamentos
manuais ou de um segundo modelo avaliador na etapa principal. A qualidade e
controlada por filtros deterministas e por rastreabilidade, deixando auditoria
humana como etapa opcional posterior.

## Fluxo geral

```text
carregamento da fonte
  -> filtro do corpus
  -> chunking dos documentos
  -> planejamento deterministico dos slots
  -> geracao batelada com LLM
  -> parse/schema do JSON gerado
  -> avaliacao deterministica
  -> novas tentativas para slots nao aceitos
  -> exportacao final com trava anti-vazamento de fonte oculta
```

Os principais artefatos de uma run ficam em `runs/<nome-da-run>/`:

- `config.resolved.json`: configuracao final usada na execucao.
- `corpus-index.jsonl`: indice dos documentos elegiveis.
- `corpus-selected.jsonl`: snapshot dos documentos selecionados e chunkados.
- `plan.json`: plano imutavel de slots de geracao.
- `run.db`: ledger SQLite com slots, tentativas, respostas e avaliacoes.
- `report.json`: resumo final da run.
- `exports/`: visoes finais do dataset.

## Corpus e filtragem inicial

A fonte usada na run principal foi o dataset Hugging Face
`costadev00/wiki-brazil`, split `train`, em modo streaming. O campo de ID foi
`__row_index__`; texto, titulo, secoes e licenca foram mapeados dos campos
`text`, `title`, `sections` e `license`.

Antes de qualquer chamada ao modelo, cada registro foi normalizado como
`Document` e passou pelo perfil `wikipedia_ptbr`. Esse perfil remove:

- documentos vazios;
- paginas fora do namespace principal;
- categorias, templates, arquivos, paginas de ajuda, portal e similares;
- redirects;
- paginas de desambiguacao;
- textos com menos de 300 caracteres.

Essa etapa e importante porque ela reduz a chance de gerar exemplos a partir de
paginas administrativas, muito curtas ou semanticamente pouco informativas.

Na run `wiki-brazil-all-tasks-22596`, o planejamento encontrou e selecionou
1.614 documentos elegiveis. Como a selecao foi `fraction=1.0`, todos os
documentos elegiveis entraram no snapshot selecionado.

## Chunking

Os documentos foram divididos em chunks de ate 8.000 caracteres, com overlap de
400 caracteres. Cada chunk vira uma secao candidata para geracao e evidencia.

Esse passo tem tres funcoes metodologicas:

1. manter o contexto de entrada controlado, mesmo em artigos longos;
2. permitir que os offsets de evidencia sejam relativos a uma secao especifica;
3. distribuir tentativas de um mesmo documento por partes diferentes do artigo.

Na run principal, os 1.614 documentos selecionados produziram 5.374 secoes
chunkadas. O texto dos documentos selecionados variou de 301 a 199.159
caracteres, com mediana de 11.667 caracteres.

## Planejamento dos slots

O planejamento nao chama o modelo. Ele apenas determina, de forma
reprodutivel, quais exemplos devem ser tentados.

Na run principal:

- seed: `42`;
- documentos selecionados: `1.614`;
- exemplos alvo: `22.596`;
- exemplos por documento: minimo `14`, maximo `14`;
- reserva de documentos: `0%`;
- tentativas por slot: ate `5`;
- limite global de tentativas: `3.0 * alvo`;
- tarefas: 14 tipos, todos com peso `1.0`;
- dificuldades: `easy=0.25`, `medium=0.5`, `hard=0.25`.

Como `22.596 = 1.614 * 14`, o plano alocou 14 slots por documento. As tarefas
foram balanceadas globalmente: cada uma das 14 tarefas recebeu exatamente 1.614
slots. Isso nao significa, porem, que cada documento recebeu exatamente uma
tarefa de cada tipo. A atribuicao documento-tarefa foi embaralhada de forma
deterministica pela seed.

Tipos de tarefa planejados:

- `classification`
- `closed_qa`
- `comparison`
- `concept_explanation`
- `definition`
- `didactic_explanation`
- `fact_checking`
- `information_extraction`
- `rewrite`
- `short_answer`
- `structured_extraction`
- `summarization`
- `taxonomy`
- `timeline`

## Geracao

A geracao foi feita com um unico modelo local:

- backend: `vllm_local`;
- modelo: `google/gemma-4-31B-it-qat-w4a16-ct`;
- `tensor_parallel_size=4`;
- `quantization=compressed-tensors`;
- `kv_cache_dtype=fp8`;
- `max_num_batched_tokens=16384`;
- janela de contexto: `65.536` tokens;
- limite de entrada: `53.248` tokens;
- limite de saida: `4.096` tokens;
- temperatura: `0.1`;
- maximo de requisicoes em voo: `32`.

Cada requisicao enviada ao modelo contem:

- prompt de sistema fixo;
- payload JSON com tipo de tarefa, lingua, dificuldade, ID do documento, titulo,
  indicador de truncamento e trecho-fonte;
- schema JSON obrigatorio para a resposta.

O schema obriga o modelo a retornar:

- `instruction`;
- `input`;
- `output`;
- `risk`;
- pelo menos um item em `evidence`, com `section_id`, `start`, `end` e,
  opcionalmente, `quote`.

O prompt instrui o modelo a usar apenas fatos presentes no trecho e a nao
mencionar o texto, documento, contexto, passagem, fonte oculta ou equivalentes.
Mesmo assim, essa instrucao do prompt nao e tratada como suficiente. Ela e
verificada depois por filtros deterministas.

## Pipeline de filtragem das geracoes

A filtragem acontece em camadas. Um candidato so chega ao dataset final se
passar por todas elas.

### 1. Parse e validacao estrutural

Primeiro, a resposta do modelo precisa ser parseavel como JSON e precisa
respeitar o schema esperado. Se o backend retorna texto invalido, JSON fora do
schema, campos ausentes ou spans impossiveis de construir como `EvidenceSpan`,
a tentativa e registrada como `error`, nao como exemplo rejeitado pelo
avaliador.

Na run principal, houve 3.180 erros desse tipo:

- 3.157 por erro de parse/schema JSON;
- 23 por erro de validacao Pydantic.

Esses casos contam como tentativas feitas, mas nao produzem candidato avaliavel.

### 2. Checagem de campos essenciais

Para candidatos validos estruturalmente, o avaliador deterministico verifica se
`instruction` e `output` nao estao vazios. Tambem rejeita instrucoes curtas
demais, abaixo de 12 caracteres.

Motivos associados:

- `empty_instruction_or_output`;
- `instruction_too_short`.

### 3. Checagem de independencia da fonte oculta

O dataset final nao deve conter comandos como "de acordo com o texto" ou "com
base no documento", porque o usuario final do exemplo SFT nao vera o documento
oculto. Por isso, a pipeline procura referencias explicitas ao texto-fonte em
campos visiveis ao usuario:

- `instruction`;
- `input`;
- `output`.

A deteccao cobre padroes em portugues, ingles e espanhol, como referencias a
texto, documento, contexto, passagem, artigo, fonte, material e equivalentes.

Motivo associado:

- `source_reference_in_candidate`.

Essa regra e aplicada duas vezes: na avaliacao deterministica e novamente antes
da exportacao. Assim, mesmo se algum exemplo problematico estivesse marcado como
aceito no banco, a exportacao seria recusada.

### 4. Checagem de duplicatas

Cada candidato e normalizado por:

- normalizacao Unicode;
- remocao de marcas combinantes;
- `casefold`;
- colapso de espacos;
- concatenacao de `instruction`, `input` e `output`.

Se a impressao digital normalizada ja existir entre os exemplos aceitos, o novo
candidato e rejeitado.

Motivo associado:

- `duplicate_candidate`.

Esse filtro evita que repeticoes quase triviais, diferencas de caixa e
variacoes de espaco entrem no dataset.

### 5. Checagem de evidencia

Cada candidato precisa apontar para pelo menos um span de evidencia em uma secao
do documento usado na tentativa. Para cada span, a pipeline verifica:

- se `section_id` existe no documento chunkado;
- se `start` e `end` estao dentro dos limites da secao;
- se `end > start`;
- se o `quote`, quando fornecido, bate exatamente com o trecho da secao;
- se o trecho recuperado tem pelo menos 20 caracteres.

Motivos associados:

- `invalid_section`;
- `invalid_evidence_offsets`;
- `evidence_quote_mismatch`;
- `missing_grounding`;
- `weak_grounding`.

Nem todo problema de evidencia causa rejeicao. `weak_grounding`, por si so, e
registrado como issue, mas nao e considerado critico. Ja secao invalida, offset
invalido, quote divergente ou ausencia completa de grounding causam rejeicao.

### 6. Decisao deterministica

O avaliador acumula todas as issues do candidato. Se qualquer issue critica
aparece, o veredito e `reject`. Caso contrario, o veredito e `accept`.

Issues criticas:

- `empty_instruction_or_output`;
- `invalid_section`;
- `invalid_evidence_offsets`;
- `evidence_quote_mismatch`;
- `missing_grounding`;
- `source_reference_in_candidate`;
- `instruction_too_short`;
- `duplicate_candidate`.

A run atual nao usa judge LLM. A funcao de roteamento para LLM sempre retorna
`False`, e o relatorio final registra `llm_judged_examples=0`,
`llm_judge_coverage=0.0` e `semantic_judge_configured=false`.

### 7. Politica de novas tentativas

Cada slot continua pendente enquanto nao houver um candidato aceito e enquanto o
numero maximo de tentativas nao tiver sido atingido. Na run principal, cada slot
podia ser tentado ate 5 vezes.

Se uma tentativa e aceita, o slot inteiro passa para `accepted`. Tentativas
posteriores ou candidatos concorrentes daquele slot sao marcados como
superseded quando necessario.

Como a run principal nao tinha documentos de reserva (`reserve_fraction=0.0`),
todas as novas tentativas continuaram usando os documentos planejados.

### 8. Trava final de exportacao

Antes de escrever os arquivos finais, o exportador reabre os candidatos aceitos
no `run.db` e roda novamente a deteccao de referencia a fonte oculta. Se
encontrar qualquer exemplo aceito com esse problema, a exportacao falha.

Essa etapa e uma garantia de ultima linha: a avaliacao faz o filtro principal,
mas a exportacao impede que exemplos fonte-dependentes escapem para os arquivos
finais.

## Resultado da run principal

Run analisada: `runs/wiki-brazil-all-tasks-22596`.

Resumo:

- status: `partial`;
- alvo: `22.596` exemplos;
- aceitos: `20.378`;
- tentativas totais: `35.876`;
- rejeitados pelo avaliador: `12.318`;
- erros de backend/parse/schema: `3.180`;
- slots pendentes ao final: `2.218`;
- taxa de aceitacao sobre tentativas: `56,8%`;
- judge LLM: `0` exemplos.

O status ficou `partial` porque 2.218 slots chegaram ao limite de 5 tentativas
sem produzir candidato aceito. Ainda assim, os artefatos exportados contem
20.378 exemplos em cada formato:

- `exports/messages/train.jsonl`;
- `exports/prompt_completion/train.jsonl`;
- `exports/alpaca/train.jsonl`.

Como os splits configurados foram `train=1.0`, `validation=0.0` e `test=0.0`,
so existe split de treino.

## Principais motivos de rejeicao

Os motivos abaixo nao sao mutuamente exclusivos: um mesmo candidato pode ter
mais de uma issue.

| Issue | Ocorrencias | Interpretacao |
| --- | ---: | --- |
| `missing_grounding` | 7.925 | nenhum span recuperavel sustentou o candidato |
| `invalid_section` | 7.890 | o `section_id` indicado nao existia no chunk usado |
| `duplicate_candidate` | 2.870 | candidato repetia conteudo ja aceito apos normalizacao |
| `source_reference_in_candidate` | 1.855 | instrucao/input/output dependiam do texto oculto |
| `weak_grounding` | 927 | evidencia recuperada era curta demais |
| `invalid_evidence_offsets` | 232 | offsets fora dos limites ou inconsistentes |
| `empty_instruction_or_output` | 16 | instrucao ou resposta vazia |
| `evidence_quote_mismatch` | 9 | quote fornecido nao batia com o span |

O par mais frequente foi `invalid_section` + `missing_grounding`, com 7.180
ocorrencias. Isso sugere que uma parte relevante das rejeicoes veio de
referencias de evidencia mal alinhadas ao chunk efetivamente usado na
tentativa, nao necessariamente de baixa qualidade textual da resposta.

## Resultado por tentativa

Aceitos por numero da tentativa:

| Tentativa | Aceitos |
| ---: | ---: |
| 1 | 17.768 |
| 2 | 1.499 |
| 3 | 616 |
| 4 | 303 |
| 5 | 192 |

A maior parte do dataset final veio da primeira tentativa. As tentativas
adicionais recuperaram mais 2.610 exemplos que teriam sido perdidos se a
pipeline tivesse usado apenas uma geracao por slot.

## Linguagem metodologica sugerida para o artigo

Uma forma concisa de descrever a pipeline:

> Construimos uma pipeline de geracao sintetica com planejamento deterministico
> e avaliacao automatica baseada em regras verificaveis. Para cada documento
> selecionado, o sistema cria slots de tarefa e dificuldade, gera candidatos
> estruturados em JSON com um modelo instrucional local e aceita apenas os
> candidatos que satisfazem restricoes de auto-contencao, nao duplicacao e
> ancoragem em spans de evidencia do texto-fonte. A avaliacao principal nao usa
> um segundo modelo julgador; todos os filtros aplicados na run analisada sao
> deterministas e auditaveis a partir do banco SQLite da execucao.

Uma versao mais detalhada para a secao de filtragem:

> A filtragem das geracoes foi organizada em camadas. Primeiro, respostas
> invalidas ou fora do schema esperado foram descartadas como erros de geracao.
> Em seguida, os candidatos estruturalmente validos foram avaliados por um
> conjunto de gates deterministas: presenca de instrucao e resposta, tamanho
> minimo da instrucao, ausencia de referencias ao contexto oculto, inexistencia
> de duplicata normalizada e validade dos spans de evidencia. Um candidato era
> rejeitado quando qualquer issue critica era detectada. Issues nao criticas,
> como evidencia curta, eram registradas para diagnostico, mas nao bastavam
> isoladamente para rejeicao. Por fim, antes da exportacao, uma trava adicional
> reavaliava os campos visiveis dos exemplos aceitos para impedir que exemplos
> dependentes do texto-fonte fossem escritos nos arquivos finais.

## Limitacoes a registrar

Alguns pontos merecem aparecer como limitacoes ou ameacas a validade:

- A avaliacao deterministica verifica forma, rastreabilidade e sinais de
  grounding, mas nao substitui uma checagem semantica completa de todas as
  afirmacoes da resposta.
- A presenca de offsets validos indica que ha evidencia recuperavel, mas nao
  garante que toda a resposta seja integralmente sustentada pelo trecho.
- A run principal terminou como `partial`; portanto, o corpus exportado contem
  20.378 exemplos aceitos, nao os 22.596 planejados.
- O filtro de duplicatas e exato apos normalizacao textual; ele nao detecta
  necessariamente parafrases semanticamente equivalentes.
- O grande volume de `invalid_section` sugere que a geracao de referencias de
  evidencia ainda pode ser melhorada, especialmente na relacao entre prompt,
  chunk selecionado e IDs de secao.

## Pontos fortes a destacar

- Planejamento deterministico por seed e hash de configuracao.
- Snapshot do corpus selecionado, evitando dependencia de mudancas posteriores
  da fonte.
- Ledger transacional em SQLite para retomar execucoes e auditar decisoes.
- Geracao e avaliacao separadas: o modelo propoe; as regras aceitam ou rejeitam.
- Exportacao com uma trava final contra vazamento de dependencia do texto
  oculto.
- Metadados de proveniencia preservados nos exemplos finais.

## Mapeamento para o codigo

Pontos principais da implementacao:

- Perfil de elegibilidade da Wikipedia: `sft_dataset_creator/profiles.py`.
- Normalizacao das fontes: `sft_dataset_creator/sources.py`.
- Chunking: `sft_dataset_creator/chunking.py`.
- Planejamento dos slots: `sft_dataset_creator/planner.py`.
- Prompts e instrucoes por tarefa: `sft_dataset_creator/prompts.py`.
- Schema e construcao de candidatos: `sft_dataset_creator/tasks.py`.
- Execucao, tentativas e retomada: `sft_dataset_creator/engine.py`.
- Ledger SQLite: `sft_dataset_creator/state.py`.
- Filtros deterministas: `sft_dataset_creator/evaluators.py`.
- Deteccao de referencia a fonte oculta e fingerprints: `sft_dataset_creator/quality.py`.
- Exportacao e trava final: `sft_dataset_creator/exporters.py`.
