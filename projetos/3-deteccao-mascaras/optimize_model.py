from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------
print ("Carregaando o modelo treinado (model.pt)")
model = YOLO("model.pt")

print ("Exportando para TensorFlow Lite")

caminho_exportado = model.export(format="tflite", imgsz = 640)

print("Arquivo está pronto na raiz do programa.")
