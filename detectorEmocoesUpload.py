import gradio as gr
import cv2
import torch
import numpy as np
from torchvision import transforms
from feat import Detector
import pandas as pd
import os
import sys

# Ajuste o caminho conforme sua estrutura de pastas
sys.path.append('/Users/klaytonramires/Documents/scripts/detecao_expressoes/Deepfake-Detection')
from network.models import model_selection

# Inicializações
detector = Detector()
deepfake_model = model_selection('xception', 2)
deepfake_model.load_state_dict(torch.load('ffpp_c40.pth', map_location='cpu'))
deepfake_model.eval()

# Dicionários de tradução e emoji
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

def get_emotion_color(emotion):
    if emotion in positive_emotions:
        return "blue"
    elif emotion in negative_emotions:
        return "red"
    else:
        return "green"

def get_veracity_color(veracity):
    return "green" if veracity == "Autêntico" else "red"

def analyze_upload(file):
    if file is None:
        return None, "<div style='padding:10px;'>Nenhum arquivo fornecido</div>", pd.DataFrame()

    path = file.name if hasattr(file, "name") else file
    ext = os.path.splitext(path)[-1].lower()

    # captura um frame com face
    frame = None
    if ext in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ok, f = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            preds = detector.detect(
                torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0),
                data_type="tensor"
            )
            if not preds.faceboxes.empty:
                frame = rgb.copy()
                break
        cap.release()
    else:
        img = cv2.imread(path)
        if img is None:
            return None, "<div style='padding:10px;'>Arquivo inválido</div>", pd.DataFrame()
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preds = detector.detect(
            torch.from_numpy(frame).permute(2,0,1).unsqueeze(0),
            data_type="tensor"
        )

    if frame is None or preds.faceboxes.empty:
        return frame, "<div style='padding:10px;'>Nenhuma face detectada</div>", pd.DataFrame()

    # ordena de cima→baixo e, dentro, esquerda→direita
    boxes = preds.faceboxes.copy()
    boxes_sorted = boxes.sort_values(by=['FaceRectY','FaceRectX']).reset_index()

    emotion_counts = {}
    result_texts = []

    for idx, row in boxes_sorted.iterrows():
        seq = idx + 1
        orig_i = row['index']
        x, y = int(row['FaceRectX']), int(row['FaceRectY'])
        w, h = int(row['FaceRectWidth']), int(row['FaceRectHeight'])
        face_crop = frame[y:y+h, x:x+w]

        # deepfake
        inp = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])(face_crop).unsqueeze(0)
        with torch.no_grad():
            out = deepfake_model(inp)
            probs = torch.nn.functional.softmax(out, dim=1)[0]
            cls = torch.argmax(probs).item()
            conf = probs[cls].item()
        veracity = "Autêntico" if cls==0 else "Possivelmente Falso"

        # emoção
        scores = preds.emotions.loc[orig_i].to_dict()
        main_em = max(scores, key=scores.get)
        p_em = scores[main_em]
        em_pt = emotion_translation[main_em]
        emoji = emotion_emoji[em_pt]
        emotion_counts[em_pt] = emotion_counts.get(em_pt, 0) + 1

        # cores e bbox
        color_box = (0,255,0) if veracity=="Autêntico" else (255,0,0)
        cv2.rectangle(frame, (x,y),(x+w,y+h), color_box, 2)

        # Número da face (fundo preto + texto branco)
        num_txt = str(seq)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_num, thick_num = 0.8, 2
        (nw, nh), nb = cv2.getTextSize(num_txt, font, scale_num, thick_num)
        cv2.rectangle(frame,
                      (x, y - nh - nb - 6),
                      (x + nw + 4, y),
                      (0,0,0), -1)
        cv2.putText(frame, num_txt,
                    (x + 2, y - nb - 2),
                    font, scale_num,
                    (255,255,255), thick_num, cv2.LINE_AA)

        # Label com emoji, emoção e veracidade (abaixo da bbox)
        lbl = f"{emoji} {em_pt} | {veracity}"
        scale_lbl, thick_lbl = 0.6, 2
        (lw, lh), lb = cv2.getTextSize(lbl, font, scale_lbl, thick_lbl)
        cv2.rectangle(frame,
                      (x, y + h),
                      (x + lw + 6, y + h + lh + lb + 6),
                      (0,0,0), -1)
        cv2.putText(frame, lbl,
                    (x + 3, y + h + lh + 2),
                    font, scale_lbl,
                    (255,255,255), thick_lbl, cv2.LINE_AA)

        # HTML numerado
        result_texts.append(f"""
        <div style='background:#fff; padding:8px; margin:4px;'>
            <b>{seq}.</b>
            <span style='color:{get_emotion_color(em_pt)};'>
                {emoji} {em_pt} ({int(p_em*100)}%)
            </span>
            &nbsp;|&nbsp;
            <span style='color:{get_veracity_color(veracity)};'>
                {veracity} ({int(conf*100)}%)
            </span>
        </div>""")

    # DataFrame gráfico
    df_plot = pd.DataFrame.from_dict(
        emotion_counts, orient='index', columns=['Contagem']
    ).reset_index().rename(columns={'index':'Emoção'})

    return frame, "<br>".join(result_texts), df_plot


# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Análise de Emoções e Veracidade")
    upload = gr.File(label="📤 Envie imagem ou vídeo")
    img_out = gr.Image(label="Frame anotado")
    html_out = gr.HTML()
    bar = gr.BarPlot(x="Emoção", y="Contagem", label="Resumo das Emoções")

    upload.change(fn=analyze_upload,
                  inputs=[upload],
                  outputs=[img_out, html_out, bar])

    gr.Markdown(
        "<center><sub>Developed by Klayton Silva Ramires. "
        "GitHub: <a href='https://github.com/kramires' target='_blank'>kramires</a></sub></center>"
    )

if __name__ == "__main__":
    demo.launch()
