"""
Funzioni di utilità condivise tra i moduli.
"""
import time
import functools
import numpy as np
import cv2


def timer(func):
    """Decorator per misurare il tempo di esecuzione."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱  {func.__name__} completato in {elapsed:.1f}s")
        return result
    return wrapper


def load_grayscale(image_path: str) -> np.ndarray:
    """Carica un'immagine in scala di grigi."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossibile caricare: {image_path}")
    return img


def binarize(img: np.ndarray, threshold: int = 160, invert: bool = True) -> np.ndarray:
    """
    Binarizza un'immagine in scala di grigi.
    Se invert=True, il testo diventa bianco (255) su sfondo nero (0).
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(img, threshold, 255, thresh_type)
    return binary


def preprocess_for_text(img: np.ndarray, threshold: int = 160) -> np.ndarray:
    """
    Preprocessa un'immagine per l'estrazione del testo:
    1. Blur gaussiano per ridurre il rumore
    2. Binarizzazione adattiva
    3. Operazioni morfologiche per pulire
    """
    # Blur leggero
    blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Binarizzazione adattiva (gestisce meglio le variazioni di illuminazione)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=10
    )

    # Rimozione rumore con apertura morfologica
    kernel_noise = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)

    return cleaned


def find_contours_sorted(binary: np.ndarray, sort_by: str = "left_to_right"):
    """
    Trova i contorni e li ordina.
    sort_by: "left_to_right", "top_to_bottom", "area"
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if sort_by == "left_to_right":
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    elif sort_by == "top_to_bottom":
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    elif sort_by == "area":
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def extract_roi(img: np.ndarray, bbox: tuple, padding: int = 0) -> np.ndarray:
    """Estrae una Region of Interest con padding opzionale."""
    x, y, w, h = bbox
    h_img, w_img = img.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)

    return img[y1:y2, x1:x2]


def normalize_glyph(glyph: np.ndarray, target_size: tuple = (32, 32)) -> np.ndarray:
    """Normalizza un glifo a una dimensione fissa mantenendo l'aspect ratio."""
    h, w = glyph.shape[:2]
    target_w, target_h = target_size

    # Calcola il fattore di scala mantenendo l'aspect ratio
    scale = min(target_w / w, target_h / h) * 0.8  # 80% per lasciare margine
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w < 1 or new_h < 1:
        return np.zeros(target_size, dtype=np.uint8)

    resized = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centra nel canvas target
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def print_header(title: str):
    """Stampa un header formattato per i log."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step: str):
    """Stampa uno step di progresso."""
    print(f"\n  -> {step}")
