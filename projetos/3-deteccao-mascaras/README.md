## 📝 Relatório do Candidato

👤 **Nome Completo:** João Vitor Lopes Miranda

### 1️⃣ Resumo da Abordagem

O treinamento foi focado na adaptação do framework ultraleve `yolo11n` utilizando o dataset fornecido. Como o treinamento foi realizado inteiramente em CPU, a estratégia adotada focou em hiperparâmetros eficientes:

- **Tamanho da Imagem (`imgsz`):** 640x640, garantindo o padrão do modelo base.
- **Épocas:** 15 épocas, o suficiente para a rede estabilizar a perda de validação dado que já partimos de pesos pré-treinados.
- **Batch Size:** 8, ajustado para não sobrecarregar a RAM/CPU durante o processo.
- **Desbalanceamento:** A classe `mask_weared_incorrect` possui pouquíssimas amostras em relação a `with_mask` e `without_mask`. Manteve-se o viés original do dataset para observar o comportamento real da rede, resultando em menor confiança da bounding box exclusivamente na classe minoritária.

### 2️⃣ Bibliotecas Utilizadas

As principais versões mapeadas durante o desenvolvimento foram:
- `ultralytics == 8.4.102`
- `torch == 2.12.1`
- `torchvision == 0.27.1`
- `litert-torch == 0.9.1`
- `ai-edge-litert == 2.1.5`
- `litert-converter == 0.2.0`

### 3️⃣ Técnica de Otimização do Modelo

A otimização foi feita exportando os pesos treinados (`model.pt`) para o formato TensorFlow Lite com quantização para INT8, usando calibração pós-treino baseada no dataset de validação:

model.export(format="tflite", imgsz=640, int8=True, data="dataset/data.yaml")

A calibração pós-treino (necessária para o INT8) usa o próprio `dataset/data.yaml` como referência de distribuição dos dados.

**Observação importante:** uma primeira tentativa de exportação sem o parâmetro `int8=True` (apenas `model.export(format="tflite", imgsz=640)`) gerou um artefato em FP32 **maior** que o `model.pt` original — a exportação sem quantização não reduz a precisão numérica dos pesos, apenas converte o formato do container (pickle/zip do PyTorch → flatbuffer do TFLite), e ainda adiciona grafos extras de NMS e metadados do delegate XNNPACK, que somam ao tamanho final. Isso não constitui otimização real para edge — foi necessário habilitar a quantização INT8 explicitamente para obter a redução de tamanho esperada em um cenário de Edge AI.

### 4️⃣ Resultados Obtidos

- **Tamanho do Arquivo Original (`model.pt`):** 5.3 MB
- **Tamanho do Arquivo Otimizado (`model.tflite`, INT8):** 3.0 MB
  (redução de 43,4% em relação ao `model.pt`)

- **mAP50 de validação — comparação FP32 (pré-otimização) vs. INT8 (final):**

| Modelo | mAP50 (all) | mAP50-95 (all) |
| :--- | :---: | :---: |
| `model.pt` (FP32, PyTorch) | 0.728 | 0.503 |
| `model.tflite` (INT8, quantizado) | 0.645 | 0.367 |

A quantização INT8 trouxe uma queda de ~11,4% no mAP50 e ~27,0% no mAP50-95 em relação ao modelo FP32 original — um trade-off esperado ao trocar precisão numérica por tamanho de arquivo menor.

- **Desempenho de Validação por Classe — `model.pt` (FP32, referência):**

| Classe | Imagens | Instâncias | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all (Geral)** | **170** | **726** | **0.784** | **0.693** | **0.728** | **0.503** |
| `with_mask` | 149 | 593 | 0.913 | 0.946 | 0.970 | 0.673 |
| `without_mask` | 57 | 114 | 0.752 | 0.711 | 0.767 | 0.498 |
| `mask_weared_incorrect` | 15 | 19 | 0.688 | 0.421 | 0.446 | 0.340 |


- **Desempenho de Validação por Classe — `model.tflite` (INT8):**

| Classe | Imagens | Instâncias | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all (Geral)** | **170** | **726** | **0.533** | **0.739** | **0.645** | **0.367** |
| `with_mask` | 149 | 593 | 0.559 | 0.946 | 0.870 | 0.487 |
| `without_mask` | 57 | 114 | 0.456 | 0.746 | 0.624 | 0.322 |
| `mask_weared_incorrect` | 15 | 19 | 0.584 | 0.526 | 0.441 | 0.291 |

Comparando com o modelo FP32 original (Precision geral de 0.784), nota-se uma **queda expressiva de Precision em todas as classes** (ex: `with_mask` caiu de 0.913 para 0.559), enquanto o Recall se manteve estável ou até melhorou levemente. Isso é consistente com um efeito colateral já identificado na etapa de inferência (seção 6️⃣): a quantização INT8 reduz a precisão numérica dos escores de confiança, tornando a etapa de NMS menos eficaz em suprimir caixas duplicadas sobre o mesmo objeto — o modelo passa a gerar mais detecções no total, incluindo falsos positivos redundantes, o que penaliza a Precision sem necessariamente perder rostos reais (Recall se mantém). A classe minoritária `mask_weared_incorrect` seguiu sendo a mais fraca em termos absolutos, como esperado dado o forte desbalanceamento do dataset (apenas 19 instâncias no conjunto de validação).

### 5️⃣ Comentários Adicionais (Opcional)

Durante o desenvolvimento, dois problemas de engenharia mereceram atenção especial:

**Conflito de dependências na exportação para TFLite:** a exportação via Ultralytics requer pacotes adicionais do ecossistema LiteRT (`litert-torch`, `ai-edge-litert`, `litert-converter`), que precisam de versões específicas e compatíveis de `torch`/`torchvision`. A solução adotada foi fixar essas versões diretamente em `requirements.txt`, garantindo que o ambiente seja determinístico tanto localmente quanto no GitHub Actions.

### 6️⃣ Exemplo de Inferência

**Saída do Terminal (`run_inference.py`, com `conf=0.35`, `iou=0.5`):**
```text
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                  9          [9x with_mask]
maksssksksss107.jpg                  1          [1x with_mask]
maksssksksss11.jpg                   24         [23x with_mask, 1x mask_weared_incorrect]
maksssksksss113.jpg                  5          [4x with_mask, 1x without_mask]
maksssksksss12.jpg                   10         [9x with_mask, 1x without_mask]
----------------------------------------------------------------------
TOTAL                                49
```

**Achado relevante do pipeline de otimização:** a primeira execução do `model.tflite` (INT8) com os parâmetros padrão de pós-processamento gerou um número de detecções visivelmente inflado em algumas imagens — por exemplo, `maksssksksss113.jpg`, uma cena com apenas 4 rostos claramente visíveis, retornou inicialmente 10 detecções. A inspeção visual das imagens anotadas em `runs/detect/inferencia_exemplos/predicoes/` confirmou caixas duplicadas sobrepostas sobre os mesmos rostos. Esse comportamento também aparece de forma quantitativa na validação formal (seção 4️⃣): a Precision caiu bastante em todas as classes no modelo INT8, enquanto o Recall se manteve — sinal de mais falsos positivos (caixas redundantes), não de rostos não detectados.

O script `run_inference.py` fornecido no repositório original foi alterado para incluir os parâmetros `conf=0.35` e `iou=0.5` na chamada de `model.predict()`, em resposta ao problema de duplicação de caixas identificado na primeira execução com os parâmetros padrão. Ajustando os parâmetros de pós-processamento na chamada de inferência, o total de detecções nas 5 imagens caiu de 81 para 49, com a contagem em `maksssksksss113.jpg` normalizando de 10 para 5 — muito mais próximo da contagem real de rostos na imagem. 

As imagens anotadas confirmam bounding boxes bem posicionadas na maioria das amostras, incluindo a identificação correta da classe minoritária (`mask_weared_incorrect`) em `maksssksksss11.jpg`. No geral, o `model.tflite` demonstrou ser funcional para uso em edge, com um trade-off mensurável de precisão em troca de um artefato ~43% menor.

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
