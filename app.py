import base64
from pathlib import Path
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from source.model import build_model
from source.gradcam import guided_gradcam, apply_heatmap


def img_to_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()

def img_tag(b64: str, size: str = "24px") -> str:
    if not b64:
        return ""
    return f'<img src="data:image/png;base64,{b64}" style="width:{size};vertical-align:middle;">'

BRAIN_B64  = img_to_b64("icons/brain.png")
DINO_B64   = img_to_b64("icons/dinosaur.png")
CLIP_B64   = img_to_b64("icons/clip.png")
METEOR_B64 = img_to_b64("icons/meteor.png")

BRAIN_IMG  = img_tag(BRAIN_B64, "28px")
DINO_IMG   = img_tag(DINO_B64,  "120px")
CLIP_IMG   = img_tag(CLIP_B64,  "16px")
METEOR_IMG = img_tag(METEOR_B64, "20px")

st.set_page_config(
    page_title="BrainScan AI",
    page_icon=f"data:image/png;base64,{BRAIN_B64}",
    layout="centered",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

  .stApp { background: #0d0d0d; color: #e0e0e0; font-family: 'Inter', sans-serif; }
  .block-container { padding-bottom: 4rem !important; }

  .header {
    text-align: center;
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1f1f1f;
    margin-bottom: 1.5rem;
  }
  .header h1 { font-size: 2rem; font-weight: 600; color: #fff; margin: 0 0 0.4rem 0; letter-spacing: 1px; }
  .header p  { color: #aaa; font-size: 0.95rem; margin: 0 0 1.2rem 0; }
  .dino-header { display: block; animation: float 3s ease-in-out infinite; }
  .dino-header img { width: 120px; }
  @keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
  }

  .msg { display: flex; gap: 12px; margin: 1rem 0; align-items: flex-start; }
  .msg.user { flex-direction: row-reverse; }
  .avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
    background: #1a1a1a; border: 1px solid #2a2a2a;
  }
  .bubble { max-width: 75%; padding: 0.75rem 1rem; border-radius: 16px; font-size: 0.9rem; line-height: 1.6; }
  .msg.user .bubble { background: #1a1a1a; border: 1px solid #333; border-top-right-radius: 4px; color: #ccc; }
  .msg.bot  .bubble { background: #161616; border: 1px solid #222; border-top-left-radius: 4px; color: #e0e0e0; }
  .bubble.danger { border-color: #ff2244; background: #1a0008; }
  .bubble.safe   { border-color: #00cc66; background: #001a0f; }

  /* File uploader */
  [data-testid="stFileUploader"] { max-width: 340px; margin-left: auto; }
  [data-testid="stFileUploader"] section {
    background: #000 !important;
    border: 2px dashed #00c9b1 !important;
    border-radius: 12px !important;
    padding: 0.5rem 0.75rem !important;
  }
  [data-testid="stFileUploader"] section p { color: #e0e0e0 !important; font-size: 0.8rem !important; }
  [data-testid="stFileUploaderDropzoneInstructions"] div span { color: #e0e0e0 !important; }
  [data-testid="stFileUploaderDropzoneInstructions"] div small { color: #aaa !important; }
  [data-testid="stFileUploader"] section button {
    background: #0a1a1a !important; color: #00c9b1 !important;
    border: 1px solid #00c9b1 !important; border-radius: 8px !important; padding: 0.4rem 1rem !important;
  }
  [data-testid="stFileUploader"] section button:hover {
    background: #0a2a2a !important; border-color: #7b61ff !important; color: #7b61ff !important;
  }
  [data-testid="stFileUploaderFile"] {
    margin-top: 10px !important; background: #1a1a1a !important;
    border: 1px solid #00c9b1 !important; border-radius: 8px !important; padding: 6px 10px !important;
  }
  [data-testid="stFileUploaderFile"] * { color: #e0e0e0 !important; }

  /* Form */
  [data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; max-width: 340px; margin-left: auto; }

  /* Launch button */
  [data-testid="stFormSubmitButton"] { display: flex !important; justify-content: flex-end !important; max-width: 340px; margin-left: auto; }
  [data-testid="stFormSubmitButton"] > button {
    width: 140px !important; min-width: 140px !important; max-width: 140px !important;
    margin-top: 8px;
    background: linear-gradient(135deg, #001a1a, #0a2a2a) !important;
    color: #00c9b1 !important; border: 1px solid #00c9b1 !important; border-radius: 12px !important;
    padding: 0.65rem 1rem !important; font-size: 1rem !important; font-weight: 600 !important;
    letter-spacing: 1px; transition: all 0.2s ease;
  }
  [data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, #0a2a2a, #1a1a3a) !important;
    box-shadow: 0 0 20px rgba(0,201,177,0.4) !important; border-color: #7b61ff !important; color: #7b61ff !important;
  }

  /* New scan button */
  div.stButton { display: flex; justify-content: flex-end; max-width: 340px; margin-left: auto; }
  div.stButton > button {
    width: 140px !important;
    background: #1a1a1a !important; color: #aaa !important;
    border: 1px solid #333 !important; border-radius: 12px !important;
    padding: 0.65rem 1rem !important; font-size: 1rem !important;
    font-weight: normal !important; letter-spacing: normal !important;
  }
  div.stButton > button:hover { border-color: #00c9b1 !important; color: #00c9b1 !important; background: #1a1a1a !important; box-shadow: none !important; }

  /* Pinned footer */
  .footer {
    position: fixed !important; bottom: 0 !important; left: 0 !important; width: 100% !important;
    text-align: center; padding: 0.6rem 0;
    background: #0d0d0d; border-top: 1px solid #1f1f1f;
    color: #aaa; font-size: 0.9rem; z-index: 9999 !important;
  }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header">
  <h1>{BRAIN_IMG} BrainScan AI</h1>
  <p>Brain tumor detection from MRI scans · EfficientNet-B0</p>
  <span class="dino-header">{DINO_IMG}</span>
</div>
""", unsafe_allow_html=True)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/efficientnet_b0.pth"

@st.cache_resource
def load_model():
    m = build_model().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "text": "Hello! I'm BrainScan AI. Upload an MRI scan and launch the meteor to analyze it.", "style": ""}
    ]
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ── Render chat + inline images ──────────────────────────
for i, msg in enumerate(st.session_state.messages):
    role_class   = "user" if msg["role"] == "user" else "bot"
    avatar       = f'<img src="data:image/png;base64,{BRAIN_B64}" style="width:20px;">' if msg["role"] == "bot" else f'<img src="data:image/png;base64,{DINO_B64}" style="width:20px;">'
    bubble_class = f"bubble {msg.get('style', '')}"
    text = msg["text"]
    # replace clip emoji with icon in user messages
    if msg["role"] == "user":
        text = text.replace("📎", CLIP_IMG)
    st.markdown(f"""
    <div class="msg {role_class}">
      <div class="avatar">{avatar}</div>
      <div class="{bubble_class}">{text}</div>
    </div>
    """, unsafe_allow_html=True)

    if "original" in msg:
        if "heatmap" in msg:
            _, c1, c2 = st.columns([1, 2, 2])
            with c1:
                st.image(msg["original"], caption="Original MRI", use_container_width=True)
            with c2:
                st.image(msg["heatmap"], caption="Grad-CAM Attention", use_container_width=True)
        else:
            _, c1, _ = st.columns([1, 2, 2])
            with c1:
                st.image(msg["original"], caption="Original MRI", use_container_width=True)

# ── Upload form ──────────────────────────────────────────
if not st.session_state.analyzed:
    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        _, btn_col = st.columns([3, 1])
        with btn_col:
            launch = st.form_submit_button("☄️ Launch", use_container_width=True)
else:
    uploaded = None
    launch = False
    _, btn_col = st.columns([3, 1])
    with btn_col:
        if st.button("🔄 New Scan", use_container_width=True):
            st.session_state.analyzed = False
            st.rerun()

if launch and uploaded:
    if st.session_state.messages[-1].get("file") != uploaded.name:
        st.session_state.messages.append({
            "role": "user",
            "text": f'{CLIP_IMG} <b>{uploaded.name}</b>',
            "style": "",
            "file": uploaded.name
        })
        img = Image.open(uploaded).convert("RGB")
        with st.spinner("Analyzing..."):
            model = load_model()
            x = tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0]
            pred_idx = probs.argmax().item()
            x_grad = tf(img).unsqueeze(0).to(DEVICE).requires_grad_(True)
            guided_cam, _ = guided_gradcam(model, x_grad, pred_idx)
            heatmap_img = apply_heatmap(img, guided_cam)

        no_tumor_pct = probs[0].item() * 100
        tumor_pct    = probs[1].item() * 100

        if pred_idx == 1:
            style = "danger"
            text  = f"⚠️ <b>Tumor Detected</b><br><br>The model identified signs of a brain tumor with <b>{tumor_pct:.1f}%</b> confidence. Please consult a medical professional."
            bot_msg = {"role": "bot", "text": text, "style": style, "original": img, "heatmap": heatmap_img}
        else:
            style = "safe"
            text  = f"✅ <b>No Tumor Detected</b><br><br>No signs of a brain tumor were found. Confidence: <b>{no_tumor_pct:.1f}%</b>."
            bot_msg = {"role": "bot", "text": text, "style": style, "original": img}

        st.session_state.messages.append(bot_msg)
        st.session_state.analyzed = True
        st.rerun()

elif launch and not uploaded:
    st.warning("Please upload an MRI scan first.")

st.markdown("<p class='footer'>For educational use only. Not a medical device.</p>", unsafe_allow_html=True)
