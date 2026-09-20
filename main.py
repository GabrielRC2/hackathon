import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Erro: não foi possível abrir a câmera.")
        return

    while True:
        success, frame = camera.read()

        if not success:
            print("Erro ao capturar frame.")
            break

        # Tamanho da imagem da câmera
        height, width = frame.shape[:2]

        # Definindo a área onde o objeto ficará
        x1 = int(width * 0.10)
        x2 = int(width * 0.90)

        y1 = int(height * 0.10)
        y2 = int(height * 0.90)

        # Recorta somente essa região
        roi = frame[y1:y2, x1:x2]

        # Desenha o retângulo na imagem original
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Texto na tela
        cv2.putText(
            frame,
            "Coloque o objeto aqui",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Mostra a câmera
        cv2.imshow("Camera", frame)

        # Mostra somente o recorte
        cv2.imshow("Recorte", roi)

        # ESC para sair
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()