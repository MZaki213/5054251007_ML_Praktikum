import numpy as np
from PIL import Image


def preprocess(img: Image.Image):
    # Gambar transparan: letakkan di atas latar putih
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)

    arr = np.array(img.convert("L"), dtype="float32")

    # Fashion-MNIST: objek terang, latar gelap. Jika tepi gambar terang, balik warnanya.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        arr = 255 - arr

    # Cari area objek
    ys, xs = np.nonzero(arr > 30)
    if len(xs) == 0:
        return None  # gambar kosong

    obj = Image.fromarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype("uint8"))

    # Skala sisi terpanjang ke 28 px (objek Fashion-MNIST memenuhi kanvas), taruh di tengah
    scale = 28 / max(obj.size)
    new_size = (max(1, round(obj.width * scale)), max(1, round(obj.height * scale)))
    obj = obj.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(obj, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    return (np.array(canvas, dtype="float32") / 255.0).reshape(1, 1, 28, 28)