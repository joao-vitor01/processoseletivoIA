import shutil

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Detecção de Máscaras Faciais (Fine-tuning do YOLO11n)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo pré-treinado YOLO11n: YOLO("yolo11n.pt")
#      (única exceção à regra de "sem modelos pré-treinados" do processo seletivo)
#   2. Fazer fine-tuning em dataset/data.yaml, em CPU (device="cpu"),
#      com um número de épocas modesto (ex: 15-30)
#   3. Copiar os pesos resultantes (results.save_dir / "weights" / "best.pt")
#      para "model.pt", na raiz desta pasta
# ---------------------------------------------------------------------------

print ("Carregando modelo YOLO11n")
model = YOLO("yolo11n.pt")

print ("Iniciando o treinamento na CPU")
results = model.train(
    data="dataset/data.yaml",
    epochs = 15,
    imgsz = 640,
    batch = 8,
    device = "cpu",
)

caminho_pesos = results.save_dir / "weights" / "best.pt"

if caminho_pesos.exists():
    shutil.copy(caminho_pesos, "model.pt")
    print("Sucesso! O arquivo 'model.pt' foi copiado para a raiz da pasta")
else:
    print("Erro: O arquivo de pesos não foi encontrado")
