import gradio as gr

from inference import CLASS_NAMES, load_model, predict_probs
from preprocess import preprocess

model = load_model()


def predict(img):
    if img is None:
        return None, None
    x = preprocess(img)
    if x is None:
        return None, None
    probs = predict_probs(model, x)
    scores = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    preview = (x[0, 0] * 255).astype("uint8")
    return scores, preview


with gr.Blocks(title="Fashion-MNIST Classifier") as demo:
    gr.Markdown("# Fashion-MNIST Classifier\nUnggah gambar satu pakaian (latar polos), lalu klik **Predict**.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Gambar pakaian")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Prediksi")
            preview = gr.Image(label="Input yang dilihat model (28x28)", image_mode="L", width=140, height=140)

    btn.click(predict, inputs=inp, outputs=[label, preview])

if __name__ == "__main__":
    demo.launch()