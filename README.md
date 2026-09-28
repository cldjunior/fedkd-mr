# FedKD-MR: aprendizagem federada heterogênea com destilação em múltiplas rodadas

Este repositório contém um notebook experimental do **FedKD-MR** (*Federated Knowledge Distillation – Multi-Round*). O método coordena clientes com arquiteturas distintas por meio das predições que seus modelos produzem sobre um conjunto público compartilhado, $D_{\mathrm{pub}}$. A coordenação ocorre no espaço de saída, sem agregação dos parâmetros dos modelos.

O notebook contempla experimentos de reconhecimento de atividades humanas com **UCI-HAR**, **Opportunity** e **WISDM**. A população heterogênea inclui modelos **MLP, CNN 1D e GRU**. Há também implementações de **FedAvg** e **FedProx** como referências homogêneas com CNN 1D.

## Proposta

Depois do treinamento supervisionado inicial, cada rodada do FedKD-MR segue estas etapas:

1. Os clientes calculam logits para as mesmas amostras de $D_{\mathrm{pub}}$.
2. Os logits formam um consenso ponderado pelo número de amostras privadas de treinamento de cada cliente.
3. Cada cliente realiza destilação de conhecimento no conjunto público usando o consenso.
4. Cada cliente realiza *fine-tuning* supervisionado com seus próprios dados privados.
5. Na rodada seguinte, os logits são recalculados a partir dos modelos atualizados.

As etapas de destilação e *fine-tuning* são sequenciais. A colaboração pressupõe uma representação de entrada compartilhada e o mesmo espaço de classes, embora os modelos possam ter arquiteturas e parametrizações diferentes. Em cada rodada, a matriz transmitida por cliente contém $|D_{\mathrm{pub}}| \times C$ valores, em que $C$ é o número de classes.

## Configurações do estudo

O artigo associado compara quatro construções do conjunto público:

| Condição | Construção de $D_{\mathrm{pub}}$ | Amostras |
|---|---|---:|
| C1 | Representativa | 512 |
| C2 | Representativa | 1.024 |
| C3 | Cobertura parcial | 512 |
| C4 | Cobertura parcial | 1.024 |

O protocolo descrito no artigo emprega **40 épocas de treinamento local inicial** e **5 rodadas** de colaboração. Cada rodada contém **8 épocas de destilação** e **8 épocas de *fine-tuning***. A temperatura de destilação é **3,0**, e o parâmetro de *sharpening* é **0,7**.

## Resultados básicos

A tabela mostra a condição com maior **acurácia média dos clientes na quinta rodada** em cada conjunto de dados, conforme o manuscrito *FedKD-MR Across IoT Sensor Domains: Public-Set Construction and Size*. Os valores são médias ± desvios padrão de **cinco execuções independentes**.

| Conjunto de dados | Melhor condição em acurácia | Acurácia média | Macro-$F_1$ |
|---|---|---:|---:|
| UCI-HAR | C2: representativa, 1.024 amostras | 0,5204 ± 0,0175 | 0,4699 ± 0,0200 |
| Opportunity | C1: representativa, 512 amostras | 0,4526 ± 0,0077 | 0,2938 ± 0,0152 |
| WISDM | C1: representativa, 512 amostras | 0,2909 ± 0,0058 | 0,2241 ± 0,0047 |

Nas construções avaliadas, a condição representativa teve maior acurácia média que a condição de cobertura parcial nos dois tamanhos e nos três conjuntos de dados. O tamanho mais favorável variou entre os domínios. Em WISDM, a acurácia mínima de um cliente permaneceu em zero em todas as condições.

Os testes globais detectaram diferenças entre C1–C4 em cada conjunto de dados. As comparações individuais entre pares não permaneceram significativas após a correção de Holm. Assim, a tabela identifica os melhores resultados descritivos dentro do protocolo avaliado, sem estabelecer superioridade estatística par a par.

## Arquivo e execução

[`FedKD_MR.ipynb`](FedKD_MR.ipynb) contém o carregamento dos conjuntos de dados, a criação de clientes e de $D_{\mathrm{pub}}$, o treinamento inicial, as rodadas de destilação e *fine-tuning*, a avaliação, os gráficos e as referências FedAvg/FedProx.

O notebook foi desenvolvido para **Google Colab** e usa caminhos do **Google Drive**. Antes de executá-lo:

1. Obtenha [UCI-HAR](https://doi.org/10.24432/C54S4K), [Opportunity](https://doi.org/10.24432/C5M027) e WISDM junto às respectivas fontes de dados.
2. Ajuste `UCI_HAR_ROOT`, `OPPORTUNITY_ROOT` e `WISDM_ROOT` na célula de carregamento.
3. Ajuste `BASE_RESULTS_DIR` para o diretório em que serão gravados métricas e artefatos.
4. Na configuração central, escolha o conjunto de dados, o modo e tamanho de $D_{\mathrm{pub}}$, o identificador do experimento e a semente.
5. Execute as células em ordem. O download dos dados e a montagem do Drive não são automatizados pelo notebook.

O notebook importa PyTorch, torchvision, NumPy, pandas, scikit-learn, Matplotlib, Seaborn e tqdm. A instalação do PyTorch deve ser adequada ao ambiente CPU ou CUDA utilizado. **Os conjuntos de dados e os artefatos de execução não estão incluídos neste repositório.**

A configuração atualmente selecionada é `UCI_HAR`, **C3** (cobertura parcial, 512 amostras), semente `42`. Essa configuração única não gera as médias de cinco execuções apresentadas acima.

## Referência

Os resultados resumidos aqui vêm do manuscrito **“FedKD-MR Across IoT Sensor Domains: Public-Set Construction and Size”**. Esse estudo examina a construção e o tamanho de $D_{\mathrm{pub}}$ em três domínios. O método FedKD-MR foi apresentado em um trabalho anterior.
