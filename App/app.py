"""

AI-USAGE DISCLAIMER:

This file was mostly done with generative AI and slightly adjusted by us.
Given that this is not a core component of the project, and mostly a visualization tool, 
we deemed unnecessary to waste time resources developing the app from scratch.

"""




import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageTk
import os
import json

# ── Paths relative to this script (safe regardless of cwd) ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _rel(filename: str) -> str:
    return os.path.join(_HERE, filename)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = _rel("KanjiOCR_2-0.keras")
INPUT_W, INPUT_H = 63, 64        # width x height fed to the model (matches training)
CANVAS_SIZE = 512                 # drawing canvas (will be downscaled)
TOP_N = 5                         # how many predictions to show
BRUSH_RADIUS = 10                 # drawing brush size (scales with canvas)

# ── Load model ───────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Could not load model: {e}")
    model = None

# ── Load kanji dictionary ─────────────────────────────────────────────────────
# JSON keys are string indices ("0", "1", ...), values are kanji characters.
# We keep int keys for easy lookup after argmax.
try:
    with open(_rel("Class2Number.json"), "r", encoding="utf-8") as f:
        _c2n: dict = json.load(f)
    Number2Letter: dict[int, str] = {int(k): v for k, v in _c2n.items()}
    print(f"✓ Dictionary loaded ({len(Number2Letter)} classes)")
except Exception as e:
    Number2Letter = {}
    print(f"✗ Could not load Class2Number.json: {e}")


def preprocess(pil_img: Image.Image) -> np.ndarray:
    """Mirror the exact preprocessing used during training."""
    img = pil_img.convert("L")                          # greyscale
    img = img.point(lambda p: 255 - p)                  # invert: black bg, white stroke
    img = img.resize((63, 64), Image.LANCZOS)           # width=63, height=64 → shape (64,63,1)
    arr = np.array(img).astype("float16") / 255.0       # float16, 0-1
    arr = arr[:, :, np.newaxis]                         # (64, 63, 1)
    arr = np.expand_dims(arr, axis=0)                   # (1, 64, 63, 1)
    return arr


def predict(pil_img: Image.Image):
    """Return list of (class_index, probability) sorted by probability desc."""
    if model is None:
        return []
    inp = preprocess(pil_img)
    probs = model.predict(inp, verbose=0)[0]         # (3036,)
    top_idx = np.argsort(probs)[::-1][:TOP_N]
    return [(int(i), float(probs[i])) for i in top_idx]



def compute_saliency(pil_img: Image.Image, class_idx: int) -> Image.Image:
    """
    Vanilla gradient saliency for class_idx.
    Returns an RGB PIL image at CANVAS_SIZE ready to blit onto the canvas.
    """
    inp_arr = preprocess(pil_img).astype("float32")   # (1, 64, 63, 1)
    img_var = tf.Variable(inp_arr)

    with tf.GradientTape() as tape:
        preds = model(img_var, training=False)
        score = preds[0, class_idx]

    grads    = tape.gradient(score, img_var)           # (1, 64, 63, 1)
    saliency = tf.abs(grads).numpy()[0, :, :, 0]      # (64, 63)

    lo, hi = saliency.min(), saliency.max()
    if hi > lo:
        saliency = (saliency - lo) / (hi - lo)

    # Original as greyscale RGB (black bg, white strokes)
    orig = np.array(
        pil_img.convert("L").point(lambda p: 255 - p)
               .resize((INPUT_W, INPUT_H), Image.LANCZOS)
    ) / 255.0
    orig_rgb = np.stack([orig] * 3, axis=-1)

    # "Hot" colourmap: black → red → yellow → white
    hot = np.zeros((*saliency.shape, 3), dtype=np.float32)
    hot[..., 0] = np.clip(saliency * 3.0,       0, 1)
    hot[..., 1] = np.clip(saliency * 3.0 - 1.0, 0, 1)
    hot[..., 2] = np.clip(saliency * 3.0 - 2.0, 0, 1)

    alpha   = saliency[:, :, np.newaxis] ** 0.5
    overlay = orig_rgb * (1 - alpha) + hot * alpha
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    result = Image.fromarray(overlay, mode="RGB")
    return result.resize((CANVAS_SIZE, CANVAS_SIZE), Image.NEAREST)


# ── GUI ───────────────────────────────────────────────────────────────────────
class KanjiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KanjiOCR")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        # PIL image used as off-screen buffer (white bg = blank)
        self._pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self._draw = ImageDraw.Draw(self._pil_img)

        self._build_ui()
        self._last_xy = None
        self._saliency_photo = None   # keeps PhotoImage reference alive
        self._last_pred_idx = None    # top predicted class for saliency

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Title
        title_f = tkfont.Font(family="Helvetica", size=18, weight="bold")
        tk.Label(self, text="KanjiOCR", font=title_f,
                 bg="#1a1a2e", fg="#e94560").pack(pady=(14, 0))
        tk.Label(self, text="Draw a kanji character",
                 bg="#1a1a2e", fg="#a0a0c0",
                 font=("Helvetica", 10)).pack(pady=(2, 8))

        # ── Horizontal body: canvas LEFT | right panel RIGHT ──────────────────
        body = tk.Frame(self, bg="#1a1a2e")
        body.pack(padx=12, pady=(0, 12))

        # Left: canvas + buttons underneath
        left = tk.Frame(body, bg="#1a1a2e")
        left.pack(side="left", anchor="n")

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="white", cursor="crosshair",
            highlightthickness=2, highlightbackground="#e94560"
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = tk.Frame(left, bg="#1a1a2e")
        btn_frame.pack(pady=(8, 0))
        self._style_btn(
            tk.Button(btn_frame, text="✦  Predict",
                      command=self._do_predict, width=12)
        ).pack(side="left", padx=6)
        self._style_btn(
            tk.Button(btn_frame, text="⌫  Clear",
                      command=self._clear, width=12),
            accent=False
        ).pack(side="left", padx=6)
        sal_btn = self._style_btn(
            tk.Button(btn_frame, text="👁  Saliency", width=12),
            accent=False
        )
        sal_btn.pack(side="left", padx=6)
        sal_btn.bind("<ButtonPress-1>",   self._saliency_press)
        sal_btn.bind("<ButtonRelease-1>", self._saliency_release)

        # Right: predictions + brush size
        right = tk.Frame(body, bg="#16213e")
        right.pack(side="left", anchor="n", padx=(12, 0), fill="y")

        # ── Predictions ───────────────────────────────────────────────────────
        tk.Label(right, text="TOP PREDICTIONS",
                 bg="#16213e", fg="#e94560",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(12, 4))

        self._result_labels = []
        for i in range(TOP_N):
            row = tk.Frame(right, bg="#16213e")
            row.pack(fill="x", padx=10, pady=4)

            tk.Label(row, text=f"#{i+1}",
                     width=3, anchor="w",
                     bg="#16213e", fg="#e94560",
                     font=("Courier", 11, "bold")).pack(side="left")

            cls_lbl = tk.Label(row, text="—",
                               width=3, anchor="w",
                               bg="#16213e", fg="#dcdcf0",
                               font=("TkDefaultFont", 22))
            cls_lbl.pack(side="left", padx=(4, 8))

            # Bar + % stacked vertically in a sub-frame
            meter = tk.Frame(row, bg="#16213e")
            meter.pack(side="left", fill="x", expand=True)

            prob_lbl = tk.Label(meter, text="",
                                anchor="w",
                                bg="#16213e", fg="#a0a0c0",
                                font=("Courier", 10))
            prob_lbl.pack(anchor="w")

            bar_canvas = tk.Canvas(meter, height=10, width=180,
                                   bg="#16213e", highlightthickness=0)
            bar_canvas.pack(anchor="w")

            self._result_labels.append((cls_lbl, bar_canvas, prob_lbl))

        # ── Brush size ────────────────────────────────────────────────────────
        tk.Frame(right, bg="#3a3a5a", height=1).pack(fill="x", padx=10, pady=(16, 0))

        tk.Label(right, text="BRUSH SIZE",
                 bg="#16213e", fg="#e94560",
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=10, pady=(10, 2))

        brush_row = tk.Frame(right, bg="#16213e")
        brush_row.pack(fill="x", padx=10, pady=(0, 12))

        self._brush_var = tk.IntVar(value=BRUSH_RADIUS)
        self._brush_lbl = tk.Label(brush_row, text=f"{BRUSH_RADIUS}px",
                                   width=5, anchor="e",
                                   bg="#16213e", fg="#dcdcf0",
                                   font=("Courier", 11))
        self._brush_lbl.pack(side="right")

        tk.Scale(
            brush_row,
            from_=2, to=40,
            orient="horizontal",
            variable=self._brush_var,
            command=self._on_brush_change,
            bg="#16213e", fg="#dcdcf0",
            troughcolor="#2a2a4a", activebackground="#e94560",
            highlightthickness=0, bd=0,
            showvalue=False,
            length=160
        ).pack(side="left", fill="x", expand=True)

        if model is None:
            self._show_error("Model not loaded — check MODEL_PATH")

    def _style_btn(self, btn, accent=True):
        btn.configure(
            bg="#e94560" if accent else "#2a2a4a",
            fg="white",
            activebackground="#c73652" if accent else "#3a3a6a",
            activeforeground="white",
            relief="flat",
            font=("Helvetica", 11, "bold"),
            pady=6,
            cursor="hand2",
            bd=0
        )
        return btn

    def _on_brush_change(self, val):
        self._brush_lbl.config(text=f"{val}px")

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _on_press(self, event):
        self._last_xy = (event.x, event.y)

    def _on_drag(self, event):
        x, y = event.x, event.y
        r = self._brush_var.get()
        if self._last_xy:
            lx, ly = self._last_xy
            # Draw on PIL buffer
            self._draw.line([lx, ly, x, y], fill=0,
                            width=r * 2)
            self._draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
            # Draw on tk canvas
            self.canvas.create_line(lx, ly, x, y,
                                    fill="black", width=r * 2,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self._last_xy = (x, y)

    def _on_release(self, _event):
        self._last_xy = None

    def _clear(self):
        self.canvas.delete("all")
        self._pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self._draw = ImageDraw.Draw(self._pil_img)
        for cls_lbl, bar_canvas, prob_lbl in self._result_labels:
            cls_lbl.config(text="—")
            bar_canvas.delete("all")
            prob_lbl.config(text="")

    # ── Prediction ────────────────────────────────────────────────────────────
    def _do_predict(self):
        results = predict(self._pil_img)
        if not results:
            self._show_error("No model loaded.")
            return
        self._last_pred_idx = results[0][0]   # save top class for saliency

        bar_w = 180   # px width for 100 %

        for i, (cls_lbl, bar_canvas, prob_lbl) in enumerate(self._result_labels):
            if i < len(results):
                cls_idx, prob = results[i]
                kanji = Number2Letter.get(cls_idx, f"#{cls_idx}")
                cls_lbl.config(text=kanji)
                prob_lbl.config(text=f"{prob*100:.1f}%")

                bar_canvas.delete("all")
                bar_canvas.config(width=bar_w)
                fill_w = int(prob * bar_w)
                alpha_hex = hex(int(0x40 + prob * 0xbf))[2:].zfill(2)
                bar_canvas.create_rectangle(
                    0, 2, bar_w, 12,
                    fill="#2a2a4a", outline=""
                )
                if fill_w > 0:
                    bar_canvas.create_rectangle(
                        0, 2, fill_w, 12,
                        fill="#e94560", outline=""
                    )
            else:
                cls_lbl.config(text="—")
                bar_canvas.delete("all")
                prob_lbl.config(text="")

    def _saliency_press(self, _event):
        if model is None or self._last_pred_idx is None:
            return
        sal_img = compute_saliency(self._pil_img, self._last_pred_idx)
        self._saliency_photo = ImageTk.PhotoImage(sal_img)
        self.canvas.create_image(0, 0, anchor="nw",
                                 image=self._saliency_photo,
                                 tags="saliency_overlay")

    def _saliency_release(self, _event):
        self.canvas.delete("saliency_overlay")
        self._saliency_photo = None

    def _show_error(self, msg: str):
        for cls_lbl, bar_canvas, prob_lbl in self._result_labels:
            cls_lbl.config(text=msg, fg="#ff6b6b")
            bar_canvas.delete("all")
            prob_lbl.config(text="")


if __name__ == "__main__":
    app = KanjiApp()
    app.mainloop()