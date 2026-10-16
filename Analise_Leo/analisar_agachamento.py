"""
Analise de agachamento com MediaPipe Pose (Tasks API).

O que o script faz:
  1. Le o video de entrada (video.mp4) frame a frame.
  2. Detecta os 33 pontos (landmarks) do corpo inteiro em cada frame.
  3. Desenha o esqueleto sobre o video e salva um video anotado.
  4. Calcula angulos articulares relevantes para o agachamento
     (joelho esq/dir, quadril esq/dir e tronco em relacao a vertical).
  5. Salva tudo (coordenadas x,y,z,visibilidade de cada landmark + angulos)
     em um CSV, um frame por linha.

Uso:
  python3 analisar_agachamento.py
  python3 analisar_agachamento.py --input video.mp4 --output-video video_analisado.mp4 --output-csv landmarks.csv

Requisitos (ja instalados neste ambiente):
  pip install mediapipe opencv-python
  Modelo baixado em: pose_landmarker_full.task
  (https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PoseLandmark = vision.PoseLandmark
PoseLandmarksConnections = vision.PoseLandmarksConnections

SCRIPT_DIR = Path(__file__).resolve().parent

# Landmarks do corpo inteiro (todos os 33 pontos do MediaPipe Pose)
LANDMARK_NAMES = [lm.name for lm in PoseLandmark if isinstance(lm.value, int) and lm.name is not None]
# Garante ordem 0..32 (o enum do MediaPipe ja segue essa ordem)
LANDMARK_NAMES = [PoseLandmark(i).name for i in range(33)]

VISIBILITY_MINIMA = 0.5  # abaixo disso, tratamos o ponto como nao confiavel para o calculo de angulos


def calcular_angulo(a, b, c):
    """Angulo (em graus) no vertice 'b', formado pelos pontos a-b-c.
    a, b, c sao tuplas (x, y) em pixels."""
    ang_ba = math.atan2(a[1] - b[1], a[0] - b[0])
    ang_bc = math.atan2(c[1] - b[1], c[0] - b[0])
    angulo = math.degrees(ang_ba - ang_bc)
    angulo = abs(angulo)
    if angulo > 180:
        angulo = 360 - angulo
    return angulo


def angulo_tronco_vertical(ombro, quadril):
    """Angulo do tronco (ombro-quadril) em relacao a vertical (0 = ereto)."""
    dx = ombro[0] - quadril[0]
    dy = ombro[1] - quadril[1]
    ang = math.degrees(math.atan2(dx, -dy))  # -dy pois y cresce para baixo na imagem
    return abs(ang)


def ponto_px(landmarks, indice, largura, altura):
    lm = landmarks[indice]
    if lm.visibility is not None and lm.visibility < VISIBILITY_MINIMA:
        return None
    return (lm.x * largura, lm.y * altura)


def main():
    parser = argparse.ArgumentParser(description="Analisa video de agachamento com MediaPipe Pose")
    parser.add_argument("--input", default=str(SCRIPT_DIR / "video.mp4"), help="Video de entrada")
    parser.add_argument("--output-video", default=str(SCRIPT_DIR / "video_analisado.mp4"), help="Video anotado de saida")
    parser.add_argument("--output-csv", default=str(SCRIPT_DIR / "landmarks_agachamento.csv"), help="CSV de saida")
    parser.add_argument("--model", default=str(SCRIPT_DIR / "pose_landmarker_full.task"), help="Modelo .task do MediaPipe")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Nao foi possivel abrir o video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (largura, altura))

    base_options = mp_python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    colunas_landmarks = []
    for nome in LANDMARK_NAMES:
        colunas_landmarks += [f"{nome}_x", f"{nome}_y", f"{nome}_z", f"{nome}_visibilidade"]

    colunas_angulos = [
        "angulo_joelho_esq", "angulo_joelho_dir",
        "angulo_quadril_esq", "angulo_quadril_dir",
        "angulo_tronco_vertical",
    ]

    csv_path = Path(args.output_csv)
    with open(csv_path, "w", newline="") as f_csv, \
         vision.PoseLandmarker.create_from_options(options) as landmarker:

        escritor = csv.writer(f_csv)
        escritor.writerow(["frame", "tempo_s"] + colunas_landmarks + colunas_angulos)

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            resultado = landmarker.detect_for_video(mp_image, timestamp_ms)

            linha_landmarks = [""] * len(colunas_landmarks)
            linha_angulos = [""] * len(colunas_angulos)

            if resultado.pose_landmarks:
                landmarks = resultado.pose_landmarks[0]

                # desenha o esqueleto no frame
                vision.drawing_utils.draw_landmarks(
                    frame,
                    landmarks,
                    connections=PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=vision.drawing_styles.get_default_pose_landmarks_style(),
                )

                # coordenadas cruas (normalizadas) de todos os 33 pontos
                for i, lm in enumerate(landmarks):
                    base = i * 4
                    linha_landmarks[base] = lm.x
                    linha_landmarks[base + 1] = lm.y
                    linha_landmarks[base + 2] = lm.z
                    linha_landmarks[base + 3] = lm.visibility

                # pontos em pixel para calculo de angulos
                ombro_e = ponto_px(landmarks, PoseLandmark.LEFT_SHOULDER.value, largura, altura)
                ombro_d = ponto_px(landmarks, PoseLandmark.RIGHT_SHOULDER.value, largura, altura)
                quadril_e = ponto_px(landmarks, PoseLandmark.LEFT_HIP.value, largura, altura)
                quadril_d = ponto_px(landmarks, PoseLandmark.RIGHT_HIP.value, largura, altura)
                joelho_e = ponto_px(landmarks, PoseLandmark.LEFT_KNEE.value, largura, altura)
                joelho_d = ponto_px(landmarks, PoseLandmark.RIGHT_KNEE.value, largura, altura)
                tornozelo_e = ponto_px(landmarks, PoseLandmark.LEFT_ANKLE.value, largura, altura)
                tornozelo_d = ponto_px(landmarks, PoseLandmark.RIGHT_ANKLE.value, largura, altura)

                ang_joelho_e = ang_joelho_d = ang_quadril_e = ang_quadril_d = ang_tronco = None

                if quadril_e and joelho_e and tornozelo_e:
                    ang_joelho_e = calcular_angulo(quadril_e, joelho_e, tornozelo_e)
                if quadril_d and joelho_d and tornozelo_d:
                    ang_joelho_d = calcular_angulo(quadril_d, joelho_d, tornozelo_d)
                if ombro_e and quadril_e and joelho_e:
                    ang_quadril_e = calcular_angulo(ombro_e, quadril_e, joelho_e)
                if ombro_d and quadril_d and joelho_d:
                    ang_quadril_d = calcular_angulo(ombro_d, quadril_d, joelho_d)
                if ombro_e and ombro_d and quadril_e and quadril_d:
                    ombro_medio = ((ombro_e[0] + ombro_d[0]) / 2, (ombro_e[1] + ombro_d[1]) / 2)
                    quadril_medio = ((quadril_e[0] + quadril_d[0]) / 2, (quadril_e[1] + quadril_d[1]) / 2)
                    ang_tronco = angulo_tronco_vertical(ombro_medio, quadril_medio)

                linha_angulos = [ang_joelho_e, ang_joelho_d, ang_quadril_e, ang_quadril_d, ang_tronco]

                # texto dos angulos no video, proximo aos joelhos
                if ang_joelho_e is not None and joelho_e:
                    cv2.putText(frame, f"{ang_joelho_e:.0f}", (int(joelho_e[0]), int(joelho_e[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if ang_joelho_d is not None and joelho_d:
                    cv2.putText(frame, f"{ang_joelho_d:.0f}", (int(joelho_d[0]), int(joelho_d[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            tempo_s = frame_idx / fps
            escritor.writerow([frame_idx, f"{tempo_s:.4f}"] + linha_landmarks + linha_angulos)

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 60 == 0 or frame_idx == total_frames:
                print(f"Processado {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()

    print(f"\nVideo anotado salvo em: {args.output_video}")
    print(f"CSV com landmarks e angulos salvo em: {csv_path}")


if __name__ == "__main__":
    main()
