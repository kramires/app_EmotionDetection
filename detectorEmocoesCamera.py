import gradio as gr
import cv2
import torch
import numpy as np
from torchvision import transforms
from feat import Detector
import pandas as pd
from datetime import datetime
import os
import sys
from collections import deque
import matplotlib.pyplot as plt
import io
from PIL import Image

# Caminho do modelo
sys.path.append('/Users/klaytonramires/Documents/scripts/detecao_expressoes/Deepfake-Detection')
from network.models import model_selection

# Inicializações
detector = Detector()
deepfake_model = model_selection('xception', 2)
deepfake_model.load_state_dict(torch.load('ffpp_c40.pth', map_location='cpu'))
deepfake_model.eval()

emotion_translation = {
    "anger": "Raiva", "disgust": "Nojo", "fear": "Medo",
    "happiness": "Felicidade", "sadness": "Tristeza",
    "surprise": "Surpresa", "neutral": "Neutro"
}

emotion_emoji = {
    "Raiva": "😠", "Nojo": "🤢", "Medo": "😨",
    "Felicidade": "😄", "Tristeza": "😢", "Surpresa": "😲", "Neutro": "😐"
}

positive_emotions = ["Felicidade", "Surpresa"]
negative_emotions = ["Raiva", "Nojo", "Medo", "Tristeza"]

log_path = "log_emocoes.csv"
emotion_counts = {}
emotion_history = deque(maxlen=5)

if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("timestamp,emotion,emotion_score,veracity,veracity_confidence\n")

def rgb_tuple_to_str(t):
    return f"rgb({t[0]},{t[1]},{t[2]})"

def get_emotion_color(emotion):
    if emotion in positive_emotions:
        return (0, 0, 255)  # Azul
    elif emotion in negative_emotions:
        return (255, 0, 0)  # Vermelho
    else:
        return (0, 255, 0)  # Verde

def get_veracity_color(veracity):
    return (0, 255, 0) if veracity == "Autêntico" else (255, 0, 0)

def plot_emotion_bar_chart():
    if not emotion_counts:
        return None
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    ax.set_title("Distribuição de Emoções")
    ax.set_xlabel("Emoção")
    ax.set_ylabel("Contagem")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def reset_app():
    global emotion_counts, emotion_history
    emotion_counts = {}
    emotion_history.clear()
    with open(log_path, "w") as f:
        f.write("timestamp,emotion,emotion_score,veracity,veracity_confidence\n")
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    df = pd.DataFrame(columns=["Emoção", "Contagem"])
    return empty_frame, "Análise resetada.", df, "", None

def analyze_frame(frame):
    global emotion_counts
    frame = frame.copy()
    if frame.dtype == np.float32:
        frame = (frame * 255).astype(np.uint8)

    tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    preds = detector.detect(tensor_frame, data_type="tensor")

    result_texts = []
    if preds.faceboxes.empty:
        df_plot = pd.DataFrame(emotion_counts.items(), columns=["Emoção", "Contagem"])
        return frame, "<div style='padding:10px;'>Nenhuma face detectada</div>", df_plot, ", ".join(emotion_history), plot_emotion_bar_chart()

    for i, det in preds.faceboxes.iterrows():
        x, y, w, h = int(det['FaceRectX']), int(det['FaceRectY']), int(det['FaceRectWidth']), int(det['FaceRectHeight'])
        conf = det['FaceScore']
        if conf < 0.95:
            continue

        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        input_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])(face_crop).unsqueeze(0)

        with torch.no_grad():
            output = deepfake_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()

        veracity = "Autêntico" if pred_class == 0 else "Possivelmente Falso"

        emotion_scores = preds.emotions.iloc[i].to_dict()
        main_emotion = max(emotion_scores, key=emotion_scores.get)
        prob = emotion_scores[main_emotion]
        emotion_pt = emotion_translation.get(main_emotion, main_emotion)
        emoji = emotion_emoji.get(emotion_pt, "")

        emotion_history.append(emotion_pt)
        emotion_counts[emotion_pt] = emotion_counts.get(emotion_pt, 0) + 1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_data = {
            "timestamp": timestamp,
            "emotion": emotion_pt,
            "emotion_score": round(prob, 3),
            "veracity": veracity,
            "veracity_confidence": round(confidence, 3)
        }
        pd.DataFrame([log_data]).to_csv(log_path, mode='a', header=False, index=False)

        color = get_emotion_color(emotion_pt)
        ver_color = get_veracity_color(veracity)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, f"{emoji} {emotion_pt} ({int(prob*100)}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{veracity} ({int(confidence*100)}%)", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ver_color, 2)

        result_html = f"""
        <div style='background-color:white; padding:10px; margin:5px;'>
            <span style='font-size:24px; color:{rgb_tuple_to_str(color)}; font-weight:bold;'>{emoji} {emotion_pt} ({int(prob*100)}%)</span>
            &nbsp;|&nbsp;
            <span style='font-size:24px; color:{rgb_tuple_to_str(ver_color)}; font-weight:bold;'>{veracity} ({int(confidence*100)}%)</span>
        </div>
        """
        result_texts.append(result_html)

    df_plot = pd.DataFrame(emotion_counts.items(), columns=["Emoção", "Contagem"])
    return frame, "<br>".join(result_texts), df_plot, ", ".join(emotion_history), plot_emotion_bar_chart()

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Detector de Emoções e Veracidade (Deepfake)")

    with gr.Row():
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        video_output = gr.Image(label="Vídeo")

    html_output = gr.HTML()

    with gr.Row():
        with gr.Column():
            emotion_table = gr.Dataframe(headers=["Emoção", "Contagem"], datatype=["str", "number"])
            emotion_history_box = gr.Textbox(label="Histórico de Emoções", interactive=False)
        emotion_graph = gr.Image(label="Gráfico de Emoções")

    download_button = gr.File(value=log_path, file_types=[".csv"], label="Baixar CSV de Emoções")
    reset_btn = gr.Button("Zerar Análise")

    webcam_input.stream(fn=analyze_frame, inputs=webcam_input,
                        outputs=[video_output, html_output, emotion_table, emotion_history_box, emotion_graph])
    reset_btn.click(fn=reset_app, inputs=None,
                    outputs=[video_output, html_output, emotion_table, emotion_history_box, emotion_graph])

    gr.Markdown("<center><sub>Developed by Klayton Silva Ramires. GitHub: <a href='https://github.com/kramires' target='_blank'>kramires</a></sub></center>")

if __name__ == "__main__":
    demo.launch()
