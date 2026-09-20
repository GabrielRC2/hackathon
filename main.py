import cv2

from visao.features import FeatureExtractor


def main():

    # Cria o extrator
    extractor = FeatureExtractor()

    # Abre a câmera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro ao abrir a câmera.")
        return

    while True:

        success, frame = camera.read()

        if not success:
            print("Erro ao capturar imagem.")
            break

        # Dimensões da imagem
        height, width = frame.shape[:2]

        # Define o ROI
        x1 = int(width * 0.25)
        x2 = int(width * 0.75)

        y1 = int(height * 0.20)
        y2 = int(height * 0.80)

        # Recorta o objeto
        roi = frame[y1:y2, x1:x2]

        # Desenha o ROI
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Coloque o objeto aqui",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera", frame)
        cv2.imshow("ROI", roi)

        # Aperte ESPACO para testar o MobileNet
        key = cv2.waitKey(1) & 0xFF

        if key == 32:

            print("\nProcessando imagem...")

            vector = extractor.extract(roi)

            print("Vetor gerado!")
            print("Dimensao:", vector.shape)

            print("Primeiros valores:")
            print(vector[:10])

        # ESC para sair
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()