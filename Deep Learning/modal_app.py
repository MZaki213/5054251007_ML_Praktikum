import modal

app = modal.App("fashion-classifier")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "gradio==6.31.0",
        "fastapi[standard]==0.162.0",
        "pillow",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_file("model.pth", "/root/model.pth")
    .add_local_file("model_def.py", "/root/model_def.py")
    .add_local_file("inference.py", "/root/inference.py")
    .add_local_file("preprocess.py", "/root/preprocess.py")
    .add_local_file("gradio_app.py", "/root/gradio_app.py")
)


@app.function(
    image=image,
    max_containers=1,       # Gradio butuh sticky session
    scaledown_window=300,   # container mati setelah 5 menit tanpa request
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    import os
    import sys

    import gradio as gr
    from fastapi import FastAPI

    sys.path.insert(0, "/root")
    os.chdir("/root")
    from gradio_app import demo

    return gr.mount_gradio_app(app=FastAPI(), blocks=demo.queue(), path="/")