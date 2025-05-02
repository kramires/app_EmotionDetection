# app_EmotionDetection
# Detector de Emoções e Veracidade (Deepfake)

Este repositório contém três ferramentas para detecção de emoções faciais e análise de veracidade (deepfake) em imagens, vídeos e webcam, utilizando Gradio e OpenCV combinados com o Py-Feat e um modelo Xception treinado em deepfakes.

## Estrutura do Projeto

```
/ (pasta raiz)
├── .gradio/
├── .venv/
├── dataset imagens/          # Exemplos ou pastas de imagens para testes
├── Deepfake-Detection/       # Repositório/classe com a estrutura do modelo Xception
├── detectorEmocoesUpload.py  # App Gradio para upload de imagem/vídeo
├── detectorEmocoesCamera.py  # App Gradio para webcam (streaming)
├── faceExpressions.py        # Script Python com detecção direta via OpenCV
├── ffpp_c23.pth              # Checkpoint do modelo deepfake (Xception)
├── ffpp_c40.pth              # Outro checkpoint compatível
├── log_emocoes.csv           # CSV de log para o app de webcam
└── log_emocoes_upload.csv    # CSV de log para app de upload
```

## Pré-requisitos

* Python 3.8 ou superior
* Dependências:

  * gradio
  * opencv-python
  * torch
  * torchvision
  * pandas
  * matplotlib
  * pillow
  * feat (Py-Feat)

## Instalação

1. Clone este repositório:

   ```bash
   git clone https://github.com/kramires/appEmotionDetection.git
   cd emotion-veracity-detector
   ```

2. Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\\Scripts\\activate  # Windows
   ```

3. Instale as dependências:

   ```bash
   pip install gradio opencv-python torch torchvision pandas matplotlib pillow feat
   ```

> **Observação:** Ajuste o caminho para `Deepfake-Detection` e os arquivos `.pth` conforme sua configuração local.

## Uso

### 1. detectorEmocoesUpload.py

App Gradio que permite o **upload** de imagens ou vídeos para análise.

```bash
python detectorEmocoesUpload.py
```

* Acesse `http://localhost:7860` no navegador.
* Envie uma imagem (`.jpg`, `.png`) ou vídeo (`.mp4`, `.avi`, `.mov`).
* O primeiro frame com face será exibido com caixas numeradas, emojis, emoções e veracidade.
* Abaixo você verá lista numerada de resultados e gráfico de distribuição de emoções.

### 2. detectorEmocoesCamera.py

App Gradio que usa a **webcam** para streaming em tempo real.

```bash
python detectorEmocoesCamera.py
```

* Acesse `http://localhost:7860` no navegador.
* Permita o uso da câmera.
* Veja o vídeo ao vivo com detecções anotadas e log resumido.
* Utilize o botão **Zerar Análise** para reiniciar contagens e histórico.
* Baixe o CSV completo de registros de emoções.

### 3. faceExpressions.py

Script Python standalone que abre uma janela OpenCV.

```bash
python faceExpressions.py
```

* Captura da câmera padrão (índice `0`).
* Exibe caixa em tempo real com emoção e pontuação.
* Pressione `q` para sair.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir *issues* ou enviar *pull requests*.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
