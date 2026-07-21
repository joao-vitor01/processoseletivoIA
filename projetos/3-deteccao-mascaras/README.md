# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** João Vitor Lopes Miranda

### 1️⃣ Resumo da Abordagem

O treinamento foi focado na adaptação do framework ultraleve YOLO11n utilizando o dataset fornecido. Como o treinamento foi realizado inteiramente em CPU, a estratégia adotada focou em hiperparâmetros eficientes:

- **Tamanho da Imagem (`imgsz`):** 640x640, garantindo o padrão do modelo base.
- **Épocas:** 15 épocas, o suficiente para a rede estabilizar a perda de validação dado que já partimos de pesos pré-treinados.
- **Batch Size:** 8, ajustado para não sobrecarregar a RAM/CPU durante o processo.
- **Desbalanceamento:** A classe `mask_weared_incorrect` possui pouquíssimas amostras em relação a `with_mask` e `without_mask`. Manteve-se o viés original do dataset para observar o comportamento real da rede, resultando em menor confiança da bounding box exclusivamente na classe minoritária.

### 2️⃣ Bibliotecas Utilizadas

O ambiente exigiu um controle rigoroso de dependências, especialmente para a exportação e inferência. As principais versões mapeadas durante o desenvolvimento foram:
- `ultralytics == 8.4.102`
- `torch == 2.12.1`
- `torchvision == 0.27.1`
- `litert-torch == 0.9.1`
- `ai-edge-litert == 2.1.5`

### 3️⃣ Técnica de Otimização do Modelo

A otimização foi realizada através da conversão dos pesos nativos do PyTorch (`model.pt`) para o formato TensorFlow Lite (`.tflite`), utilizando a engine unificada do Google LiteRT.
O comando `model.export(format="tflite", imgsz=640)` engatilha uma pipeline complexa *under-the-hood*:

1. Os grafos do PyTorch foram inicialmente rebaixados para abstrações `FX`.
2. Em seguida, foram convertidos para a representação intermediária `MLIR`.
3. Finalmente, os módulos MLIR foram compilados para o binário `.tflite`.

Para garantir que a arquitetura gerasse um modelo perfeitamente funcional (evitando falha de CI/CD) utilizou-se o Docker no processo inicial de validação local.


### 4️⃣ Resultados Obtidos

- **Tamanho do Arquivo Original (`model.pt`):** 5.2 MB
- **Tamanho do Arquivo Otimizado (`model.tflite`):** 10.1 MB. O modelo TFLite gerado pela Ultralytics embute operações do XNNPACK e grafos adicionais para NMS (Non-Maximum Suppression), o que resulta em um artefato ligeiramente maior, porém altamente veloz na inferência.

- **Desempenho de Validação por Classe:**

| Classe | Imagens (class) | Instâncias (boxes) | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all (Geral)** | **170** | **726** | **0.784** | **0.693** | **0.728** | **0.503** |
| `with_mask` | 149 | 593 | 0.913 | 0.946 | 0.970 | 0.673 |
| `without_mask` | 57 | 114 | 0.752 | 0.711 | 0.767 | 0.498 |
| `mask_weared_incorrect` | 15 | 19 | 0.688 | 0.421 | 0.446 | 0.340 |

As classes `with_mask` e `without_mask` apresentaram excelente desempenho, com destaque para a precisão e mAP da classe correta. Conforme previsto devido ao desbalanceamento crônico do dataset, a classe minoritária `mask_weared_incorrect` sofreu uma queda expressiva no Recall (0.421) e no mAP (0.446), refletindo a dificuldade do modelo em generalizar padrões para amostras escasas.

### 5️⃣ Comentários Adicionais (Opcional)

O maior desafio deste projeto não foi o treinamento do YOLO em si, mas a orquestração do ambiente de otimização e inferência em pipelines contínuos (GitHub Actions).
Durante a exportação para `.tflite`, a Ultralytics realiza o download dinâmico do pacote `litert-torch`, o qual realiza um *downgrade* forçado do PyTorch (para a versão `2.12.1`), mas falha em realinhar o `torchvision` equivalente.
Isso causa a quebra imediata do script `run_inference.py` por incompatibilidade no binário.
**Solução:** Implementei um realinhamento no próprio `optimize_model.py` utilizando o comando `os.system("pip install litert-torch ai-edge-litert torchvision")` para garantir a sincronia e o pareamento das dependências (`torch 2.12.1` e `torchvision 0.27.1`). Isso permitiu que a integração contínua finalizasse com sucesso a inferência.


### 6️⃣ Exemplo de Inferência

**Saída do Terminal (`run_inference.py`):**
```text
Loading /workspaces/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                  9          [9x with_mask]
maksssksksss107.jpg                  1          [1x with_mask]
maksssksksss11.jpg                   23         [22x with_mask, 1x mask_weared_incorrect]
maksssksksss113.jpg                  4          [3x with_mask, 1x without_mask]
maksssksksss12.jpg                   13         [11x with_mask, 2x without_mask]
----------------------------------------------------------------------
TOTAL                                50
```
A execução do artefato de Edge ('model.tflite') demonstrou excelente robustez na prática. As imagens individuais geradas na pasta 'runs/detect/inferencia_exemplos/predicoes/' mostram que as bounding boxes foram delimitadas de maneira precisa e correta na grande maioria das amostras.

Observou-se espeficamente na imagem 'maksssksksss11.jpg' (que conteve um alto volume de detecções simultâneas) que o modelo performou muito bem na marcação geral dos rostos,  conseguindo inclusive identificar a classe minoritária (mask_weared_incorrect). No geral, otimizado provou estar pronto para cenários reais de edge, mantendo alta fidelidade com os pesos originais do PyTorch.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
