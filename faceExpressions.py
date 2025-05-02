# Variáveis para otimização de processamento
import cv2
import numpy as np
import time
from feat import Detector, Fex
import torch
from torchvision import transforms
import sys
import pandas as pd
from PIL import Image
# Adiciona o repositório Deepfake-Detection ao PYTHONPATH
sys.path.append('/Users/klaytonramires/Documents/scripts/detecao_expressoes/Deepfake-Detection')
from network.models import model_selection

# Inicializar o detector de expressões faciais com configurações para detectar AUs e emoções
detector = Detector()

# Dicionário para tradução das emoções para o português
emotion_translation = {
    "anger": "Raiva",
    "disgust": "Nojo",
    "fear": "Medo",
    "happiness": "Felicidade",
    "sadness": "Tristeza",
    "surprise": "Surpresa",
    "neutral": "Neutro"
}

 # Inicializar o modelo XceptionNet via model_selection com número de classes de saída
deepfake_model = model_selection('xception', 2)
# Carrega state_dict do Xception pré-treinado
state_dict = torch.load('ffpp_c23.pth', map_location='cpu', weights_only=False)
deepfake_model.load_state_dict(state_dict)
deepfake_model.eval()

# Transformações para deepfake
deepfake_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

 # Variável para status anterior de deepfake
prev_status = None
# Estado anterior para controle de logs
prev_has_face = None
prev_emotion_label = None

# Variáveis para otimização de processamento
frame_count = 0
process_every_n_frames = 1
confidence_threshold = 0.95
face_resize_dim = (128, 128)

# Iniciar a captura de vídeo (backend automático)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não foi possível abrir a câmera. Verifique permissões.")
    exit(1)
print("Câmera aberta no índice 0")

# Aguardar inicialização da câmera
time.sleep(1)
# Descartar alguns frames iniciais
for _ in range(5):
    cap.read()

while True:
    # Capturar frame por frame
    ret, frame = cap.read()
    if not ret:
        break
    # Controle de processamento: processa apenas a cada N frames
    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        # Exibe o frame mesmo quando não processado
        cv2.imshow('Face Expression and AU Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Conversão BGR para RGB para compatibilizar com Py-Feat
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Converte frame RGB (numpy array) para tensor PyTorch com batch dimension
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0)
    preds = detector.detect(frame_tensor, data_type="tensor")
    # Extrai as caixas de face detectadas (DataFrame Fex.faceboxes)
    faceboxes = preds.faceboxes
    # Se não houver faces, exibe o frame e continua
    if faceboxes.empty:
        cv2.imshow('Face Expression and AU Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Itera sobre cada detecção válida
    for _, det in faceboxes.iterrows():
        x1, y1 = det['FaceRectX'], det['FaceRectY']
        w, h = det['FaceRectWidth'], det['FaceRectHeight']
        conf = det['FaceScore']
        if conf < confidence_threshold:
            continue
        # Converte para inteiros
        x, y, w, h = int(x1), int(y1), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Reaproveita o resultado já calculado em preds para landmarks, emoções e AUs
        # preds contém colunas .au e .emotions; se preferir, use detect_landmarks etc.
        emotions = preds.emotions.iloc[_].to_dict()
        aus = preds.aus.iloc[_].to_dict()

        # Exibe emoção e AUs
        main_emotion = max(emotions, key=emotions.get)
        if main_emotion != prev_emotion_label:
            print(f"Emoção: {emotion_translation.get(main_emotion, main_emotion)} ({emotions[main_emotion]:.2f})")
            prev_emotion_label = main_emotion
        emotion_text = f"{emotion_translation.get(main_emotion, main_emotion)}: {emotions[main_emotion]:.2f}"
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        au_text = ', '.join([f"AU{au}: {score:.2f}" for au, score in aus.items()])
        cv2.putText(frame, au_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)



    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando tudo estiver feito, liberar a captura
cap.release()
cv2.destroyAllWindows()
