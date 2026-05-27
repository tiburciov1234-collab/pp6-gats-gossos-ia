import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import h5py
import json
import os

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def conv2d(x, w, b):
    H, W, _ = x.shape
    kH, kW, _, C_out = w.shape
    out = np.zeros((H-kH+1, W-kW+1, C_out))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(C_out):
                out[i,j,k] = np.sum(x[i:i+kH, j:j+kW, :] * w[:,:,:,k]) + b[k]
    return out

def maxpool(x):
    H, W, C = x.shape
    out = np.zeros((H//2, W//2, C))
    for i in range(H//2):
        for j in range(W//2):
            out[i,j,:] = np.max(x[i*2:i*2+2, j*2:j*2+2, :], axis=(0,1))
    return out

def predict_image(img_array, weights_path):
    with h5py.File(weights_path, 'r') as f:
        def get_w(path): return f[path][()]
        try:
            c1w = get_w('layers/conv2d/vars/0'); c1b = get_w('layers/conv2d/vars/1')
            c2w = get_w('layers/conv2d_1/vars/0'); c2b = get_w('layers/conv2d_1/vars/1')
            d1w = get_w('layers/dense/vars/0'); d1b = get_w('layers/dense/vars/1')
            d2w = get_w('layers/dense_1/vars/0'); d2b = get_w('layers/dense_1/vars/1')
        except:
            c1w = get_w('conv2d/conv2d/kernel:0'); c1b = get_w('conv2d/conv2d/bias:0')
            c2w = get_w('conv2d_1/conv2d_1/kernel:0'); c2b = get_w('conv2d_1/conv2d_1/bias:0')
            d1w = get_w('dense/dense/kernel:0'); d1b = get_w('dense/dense/bias:0')
            d2w = get_w('dense_1/dense_1/kernel:0'); d2b = get_w('dense_1/dense_1/bias:0')
    x = img_array[0]
    x = relu(conv2d(x, c1w, c1b))
    x = maxpool(x)
    x = relu(conv2d(x, c2w, c2b))
    x = maxpool(x)
    x = x.flatten()
    x = relu(x @ d1w + d1b)
    x = sigmoid(x @ d2w + d2b)
    return float(x[0])

st.set_page_config(page_title="Classificador Gats vs Gossos", layout="centered")
st.title("Classificador de Gossos i Gats")
st.markdown("Puja una imatge i la IA intentarà dir si veu un gos o un gat.")

uploaded_file = st.file_uploader("Puja una imatge en format JPG o PNG", type=["jpg","jpeg","png"])

if not os.path.exists("model_gats_gossos.weights.h5"):
    st.error("No s'ha trobat el fitxer de pesos del model.")
elif uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imatge pujada", use_container_width=True)
        img_array = np.array(image.resize((100,100))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        with st.spinner("Analitzant..."):
            prob = predict_image(img_array, "model_gats_gossos.weights.h5")
        if prob > 0.5:
            st.success(f"La IA creu que és un GOS amb un {prob*100:.2f}% de confiança.")
        else:
            st.success(f"La IA creu que és un GAT amb un {(1-prob)*100:.2f}% de confiança.")
        st.info("Recorda: el model és senzill i pot equivocar-se.")
    except UnidentifiedImageError:
        st.error("No s'ha pogut llegir la imatge.")
    except Exception as e:
        st.error(f"Error: {e}")
