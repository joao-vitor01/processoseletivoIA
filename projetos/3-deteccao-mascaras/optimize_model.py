import os
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Otimização do Modelo (Exportação para Edge)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.pt"
#   2. Exportar para TensorFlow Lite via model.export(format="tflite")
#      (a Ultralytics gera automaticamente "model.tflite" na mesma pasta)
# ---------------------------------------------------------------------------
def main():
    print ("Carregando o modelo treinado (model.pt)")
    model_path = "model.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"O arquivo {model_path} não foi encontrado na raiz do projeto.")
    model = YOLO(model_path)
    print("Iniciando processo de exportação e otimização")

    try:
        caminho_exportado = model.export(format="tflite", imgsz = 640)
        print(f"Sucesso na exportação TFLite: {caminho_exportado}")
    except Exception as e:
        print(f"\nFalha na exportação nativa: {e} ")

        if os.path.exists("model.tflite"):
            print("Artefato 'model.tflite' valido!")
        else:
            print("Tentando via ONNX")
            try:
                model.export(format="onnx", imgsz=640)
            except Exception as e_onnx:
                print(f"Erro crítico na exportação: {e_onnx}")
                raise e_onnx
if __name__ == "__main__":
    main()

