"""Interface Streamlit do projeto de Processamento Digital de Imagens."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdi import operations as op  # noqa: E402
from pdi.io import load_png, to_png_bytes  # noqa: E402


st.set_page_config(page_title="Processamento Digital de Imagens", page_icon="🖼️", layout="wide")
st.title("Processamento Digital de Imagens")
st.caption("Carregue um PNG L (8 bits) ou RGB (24 bits), configure a operação e compare os resultados.")

uploaded = st.sidebar.file_uploader("Imagem de entrada", type=["png"])
if uploaded is None:
    st.info("Carregue uma imagem PNG pela barra lateral para começar.")
    st.stop()

try:
    image, mode = load_png(uploaded)
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

is_rgb = mode == "RGB"
st.sidebar.success(f"{mode} · {image.shape[1]} × {image.shape[0]} px")

all_operations = [
    "Decomposição RGB", "Decomposição HSV", "Limiarização",
    "Transformação logarítmica", "Transformação de potência",
    "Equalização de histograma", "Fatiamento por intensidade",
    "Média gaussiana", "Filtro de ordem", "Mediana adaptativa",
    "Máscara de aguçamento", "Realce por Laplaciano", "Gradiente de Sobel",
    "Filtro em frequência", "Ruído gaussiano", "Ruído impulsivo",
]
incompatible = {"Equalização de histograma"} if is_rgb else {"Decomposição RGB", "Decomposição HSV"}
available = [name for name in all_operations if name not in incompatible]
operation = st.sidebar.selectbox("Operação", available)
if incompatible:
    st.sidebar.caption("Indisponíveis neste modo: " + ", ".join(sorted(incompatible)))


def odd_window(label: str = "Tamanho da janela", maximum: int = 15, value: int = 3) -> int:
    return st.sidebar.slider(label, 3, maximum, value, step=2)


outputs: dict[str, np.ndarray] = {}
histograms: tuple[np.ndarray, np.ndarray] | None = None

if operation == "Decomposição RGB":
    outputs = op.decompose_rgb(image)
elif operation == "Decomposição HSV":
    outputs = op.decompose_hsv(image)
elif operation == "Limiarização":
    k = st.sidebar.slider("Limiar k", 0, 255, 127)
    outputs = {"Limiarizada": op.threshold(image, k)}
elif operation == "Transformação logarítmica":
    gain = st.sidebar.slider("Ganho c", 0.1, 3.0, 1.0, 0.1)
    outputs = {"Logarítmica": op.logarithmic(image, gain)}
elif operation == "Transformação de potência":
    gain = st.sidebar.slider("Ganho c", 0.1, 3.0, 1.0, 0.1)
    gamma = st.sidebar.slider("Gama γ", 0.1, 5.0, 1.0, 0.1)
    outputs = {"Potência": op.power_law(image, gain, gamma)}
elif operation == "Equalização de histograma":
    result, before, after = op.equalize_histogram(image)
    outputs = {"Equalizada": result}
    histograms = before, after
elif operation == "Fatiamento por intensidade":
    low, high = st.sidebar.slider("Faixa [A, B]", 0, 255, (80, 180))
    preserve = st.sidebar.checkbox("Preservar fundo", value=True)
    outputs = {"Fatiamento": op.intensity_slice(image, low, high, preserve)}
elif operation == "Média gaussiana":
    size = odd_window()
    sigma = st.sidebar.slider("Desvio padrão σ", 0.1, 10.0, 1.0, 0.1)
    outputs = {"Suavizada": op.gaussian_mean(image, size, sigma)}
elif operation == "Filtro de ordem":
    kind = st.sidebar.selectbox("Tipo", ["Mediana", "Mínimo", "Máximo"])
    size = odd_window()
    outputs = {kind: op.order_filter(image, size, kind)}
elif operation == "Mediana adaptativa":
    size = odd_window("Tamanho máximo da janela")
    outputs = {"Mediana adaptativa": op.adaptive_median(image, size)}
elif operation == "Máscara de aguçamento":
    size = odd_window()
    sigma = st.sidebar.slider("Desvio padrão da máscara", 0.1, 10.0, 1.0, 0.1)
    gain = st.sidebar.slider("Ganho", 0.0, 5.0, 1.0, 0.1)
    outputs = {"Aguçada": op.unsharp_mask(image, size, sigma, gain)}
elif operation == "Realce por Laplaciano":
    size = odd_window()
    gain = st.sidebar.slider("Ganho", 0.0, 5.0, 1.0, 0.1)
    enhanced, laplacian = op.laplacian_enhance(image, size, gain)
    outputs = {"Realçada": enhanced, "Laplaciano ajustado": laplacian}
elif operation == "Gradiente de Sobel":
    size = odd_window()
    outputs = {"Magnitude do gradiente": op.sobel_gradient(image, size)}
elif operation == "Filtro em frequência":
    family = st.sidebar.selectbox("Família", ["Gaussiano", "Butterworth"])
    pass_type = st.sidebar.selectbox("Tipo", ["Passa-baixa", "Passa-alta"])
    max_cutoff = max(1.0, np.hypot(*image.shape[:2]) / 2)
    cutoff = st.sidebar.slider("Frequência de corte", 1.0, float(max_cutoff), min(30.0, float(max_cutoff)))
    order = st.sidebar.slider("Ordem", 1, 10, 2, disabled=family != "Butterworth")
    outputs = {f"{pass_type} {family}": op.frequency_filter(image, family, pass_type, cutoff, order)}
elif operation == "Ruído gaussiano":
    mean = st.sidebar.slider("Média da distribuição", -100.0, 100.0, 0.0, 1.0)
    sigma = st.sidebar.slider("Intensidade (desvio padrão)", 0.0, 100.0, 20.0, 1.0)
    seed = st.sidebar.number_input("Semente", 0, 999999, 42)
    outputs = {"Com ruído gaussiano": op.gaussian_noise(image, mean, sigma, int(seed))}
elif operation == "Ruído impulsivo":
    kind = st.sidebar.selectbox("Tipo", ["Sal", "Pimenta", "Sal e pimenta"])
    amount = st.sidebar.slider("Proporção de pixels", 0.0, 1.0, 0.05, 0.01)
    ratio = st.sidebar.slider("Fração de sal na distribuição", 0.0, 1.0, 0.5, 0.05,
                              disabled=kind != "Sal e pimenta")
    seed = st.sidebar.number_input("Semente", 0, 999999, 42)
    outputs = {f"Ruído {kind.lower()}": op.impulse_noise(image, kind, amount, ratio, int(seed))}

columns = st.columns(min(4, len(outputs) + 1))
with columns[0]:
    st.subheader("Original")
    st.image(image, clamp=True, use_container_width=True)
for index, (name, result) in enumerate(outputs.items(), start=1):
    with columns[index % len(columns)]:
        st.subheader(name)
        st.image(result, clamp=True, use_container_width=True)
        st.download_button(
            f"Baixar {name}.png", to_png_bytes(result),
            file_name=f"{name.lower().replace(' ', '_')}.png", mime="image/png",
            key=f"download-{index}", use_container_width=True,
        )

if histograms:
    st.subheader("Histogramas de intensidade")
    figure, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
    for axis, values, title in zip(axes, histograms, ("Antes", "Depois")):
        axis.bar(np.arange(256), values, width=1, color="#2563eb")
        axis.set(title=title, xlabel="Intensidade", ylabel="Pixels", xlim=(0, 255))
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

with st.expander("Sobre a operação"):
    st.write(
        "Todas as operações partem sempre da imagem original carregada. "
        "Filtros espaciais usam extensão refletida nas bordas; filtros em frequência "
        "usam a transformada 2-D por canal."
    )

