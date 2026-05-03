# =============================================================
# Pokémon Image Cleaning GUI
# Pokemon Image Classifier — Data Pipeline
#
# USAGE: Run as a standalone script, not inside Jupyter.
#        python cleaning_gui.py
#
# SETUP: Update the two paths in the CONFIG section below
#        to point to your local or Google Drive folders.
#
# REQUIREMENTS: pip install rembg pillow numpy scipy
# =============================================================

import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from pathlib import Path
import threading
import shutil
from collections import deque
from rembg import remove

# =============================================================
# CONFIG — update these two paths before running
# =============================================================
COMPLETED_ROOT = Path(r"G:\My Drive\Machine Learning Group Project\Completed Samples")
PROCESSED_ROOT = Path(r"G:\My Drive\Machine Learning Group Project\Processed Images")

# Pokédex range (Gen 1 only)
POKEDEX_START = "001"
POKEDEX_END   = "151"

# Image settings
TARGET_SIZE  = 256
REQUIRED     = 40
BRUSH_SIZE   = 10
ZOOM_STEP    = 1.25
MAX_ZOOM     = 8.0
MIN_ZOOM     = 1.0
TOLERANCE    = 20
MAX_DISPLAY  = 500
IMAGE_EXTS   = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# =============================================================
# SECTION 1 — File Helpers
# =============================================================
def get_processed_count(folder_name: str) -> int:
    """Return number of processed images already saved for a Pokémon folder."""
    processed_dir = PROCESSED_ROOT / folder_name
    if not processed_dir.exists():
        return 0
    return sum(
        1 for p in processed_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

def get_final_filename(folder_name: str) -> str:
    """Return next available Final XX.png path in the processed folder."""
    dest_dir = PROCESSED_ROOT / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        candidate = dest_dir / f"Final {i:02d}.png"
        if not candidate.exists():
            return str(candidate)
        i += 1

def resize_to_256(img: Image.Image) -> Image.Image:
    """Resize image to TARGET_SIZE x TARGET_SIZE preserving RGBA."""
    img = img.convert("RGBA")
    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    return img

# =============================================================
# SECTION 2 — Queue Setup
# =============================================================
# Auto-delete completed folders from Completed Samples
for f in sorted(COMPLETED_ROOT.iterdir()):
    if not f.is_dir():
        continue
    folder_id = f.name.split(" - ")[0].strip()
    if not (POKEDEX_START <= folder_id <= POKEDEX_END):
        continue
    if get_processed_count(f.name) >= REQUIRED:
        shutil.rmtree(f)
        print(f"  🗑  Deleted completed folder: {f.name}")

# Sort folders by gap ascending (smallest gap = closest to done first)
all_folders_with_gaps = []
for f in sorted(COMPLETED_ROOT.iterdir()):
    if not f.is_dir():
        continue
    folder_id = f.name.split(" - ")[0].strip()
    if not (POKEDEX_START <= folder_id <= POKEDEX_END):
        continue
    count = get_processed_count(f.name)
    if count >= REQUIRED:
        print(f"  ✓ Skipping {f.name} ({count}/{REQUIRED} complete)")
        continue
    gap = REQUIRED - count
    all_folders_with_gaps.append((gap, f))

all_folders_with_gaps.sort(key=lambda x: x[0])

all_files = []
for gap, folder in all_folders_with_gaps:
    print(f"  → {folder.name}: needs {gap} more")
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            all_files.append(p)

total_needed = sum(gap for gap, _ in all_folders_with_gaps)
print(f"\nImages in queue: {len(all_files):,}")
print(f"Total still needed: {total_needed:,}")
print(f"Incomplete Pokémon: {len(all_folders_with_gaps):,}")

if not all_files:
    print("All Pokémon complete!")
    raise SystemExit(0)

queue = list(all_files)
current_index = 0

# =============================================================
# SECTION 3 — Image Processing Helpers
# =============================================================
def flood_fill_mask(img_array: np.ndarray, x: int, y: int,
                    tolerance: int) -> np.ndarray:
    """Magic wand flood fill — returns boolean mask of selected region."""
    h, w = img_array.shape[:2]
    target = img_array[y, x, :3].astype(int)
    visited = np.zeros((h, w), dtype=bool)
    mask    = np.zeros((h, w), dtype=bool)
    stack   = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        pixel = img_array[cy, cx, :3].astype(int)
        if np.max(np.abs(pixel - target)) <= tolerance:
            mask[cy, cx] = True
            stack.extend([(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)])
    return mask

def color_range_mask(img_array: np.ndarray, x: int, y: int,
                     tolerance: int) -> np.ndarray:
    """Select all pixels globally similar to the clicked color."""
    target = img_array[y, x, :3].astype(int)
    diff   = np.max(np.abs(img_array[:, :, :3].astype(int) - target), axis=2)
    return diff <= tolerance

# =============================================================
# SECTION 4 — Main GUI Application
# =============================================================
class ImageReviewApp:
    def __init__(self, root: tk.Tk):
        self.root        = root
        self.root.title("Pokémon Image Review Tool")
        self.queue       = queue
        self.index       = 0
        self.zoom        = 1.0
        self.tool        = "magic_wand"
        self.orig_img    = None   # PIL Image before any edits
        self.current_img = None   # PIL Image with edits applied
        self.pre_rembg   = None   # PIL Image saved before rembg
        self.undo_stack  = deque(maxlen=10)
        self.rembg_running = False
        self._build_ui()
        self._bind_keys()
        self.load_image()

    # ── UI Construction ──────────────────────────────────────
    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, pady=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="👁 Preview",  command=self.preview_image).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="💾 Keep",     command=self.keep_image).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="🪄 Remove BG",command=self.remove_background).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="⏭ Skip",      command=self.skip_image).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="↩ Undo",      command=self.undo).pack(side=tk.LEFT, padx=4)

        # Tool selector
        tool_frame = tk.LabelFrame(toolbar, text="Tool", padx=4)
        tool_frame.pack(side=tk.LEFT, padx=8)
        for tool, label in [
            ("magic_wand",   "🪄 Magic Wand"),
            ("color_range",  "🎨 Color Range"),
            ("restore",      "✏️ Restore"),
            ("erase",        "⬜ Erase"),
        ]:
            tk.Radiobutton(
                tool_frame, text=label, value=tool,
                variable=tk.StringVar(value=self.tool),
                command=lambda t=tool: self.set_tool(t)
            ).pack(side=tk.LEFT)

        # Progress bar
        self.progress_var = tk.StringVar(value="")
        tk.Label(toolbar, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=8)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#2b2b2b",
                                width=MAX_DISPLAY, height=MAX_DISPLAY,
                                cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 anchor=tk.W, relief=tk.SUNKEN).pack(
            side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        self.canvas.bind("<Button-1>",       self.on_click)
        self.canvas.bind("<B1-Motion>",      self.on_drag)
        self.canvas.bind("<MouseWheel>",     self.on_scroll)
        self.canvas.bind("<Button-4>",       self.on_scroll)
        self.canvas.bind("<Button-5>",       self.on_scroll)
        self.root.bind("<Return>",           lambda e: self.keep_image())
        self.root.bind("<space>",            lambda e: self.skip_image())
        self.root.bind("<Control-z>",        lambda e: self.undo())

    # ── Image Loading ─────────────────────────────────────────
    def load_image(self):
        if self.index >= len(self.queue):
            self.status_var.set("✅ Queue complete!")
            return
        path = self.queue[self.index]
        try:
            img = Image.open(path).convert("RGBA")
            self.orig_img    = img.copy()
            self.current_img = img.copy()
            self.pre_rembg   = None
            self.undo_stack.clear()
            self.zoom = 1.0
            self._update_progress(path)
            self._render()
        except Exception as e:
            self.status_var.set(f"Error loading image: {e}")
            self.skip_image()

    def _update_progress(self, path: Path):
        folder_name  = path.parent.name
        processed    = get_processed_count(folder_name)
        self.progress_var.set(
            f"{folder_name} — {processed}/{REQUIRED} done | "
            f"Queue: {self.index+1}/{len(self.queue)}"
        )

    # ── Rendering ─────────────────────────────────────────────
    def _render(self):
        if self.current_img is None:
            return
        w = int(self.current_img.width  * self.zoom)
        h = int(self.current_img.height * self.zoom)
        display = self.current_img.resize((w, h), Image.NEAREST)

        # Checkerboard background for transparency
        checker = Image.new("RGBA", (w, h), (200, 200, 200, 255))
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                if (x // 10 + y // 10) % 2 == 0:
                    for dy in range(min(10, h-y)):
                        for dx in range(min(10, w-x)):
                            checker.putpixel((x+dx, y+dy), (255, 255, 255, 255))
        checker.paste(display, mask=display.split()[3])

        self._tk_img = ImageTk.PhotoImage(checker)
        self.canvas.config(width=min(w, MAX_DISPLAY), height=min(h, MAX_DISPLAY))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

    # ── Tool Interactions ─────────────────────────────────────
    def set_tool(self, tool: str):
        self.tool = tool
        self.status_var.set(f"Tool: {tool}")

    def on_click(self, event):
        self._apply_tool(event.x, event.y)

    def on_drag(self, event):
        if self.tool in ("restore", "erase"):
            self._apply_tool(event.x, event.y)

    def _apply_tool(self, canvas_x: int, canvas_y: int):
        if self.current_img is None:
            return
        img_x = int(canvas_x / self.zoom)
        img_y = int(canvas_y / self.zoom)
        w, h  = self.current_img.size
        if not (0 <= img_x < w and 0 <= img_y < h):
            return

        self.undo_stack.append(self.current_img.copy())
        arr = np.array(self.current_img)

        if self.tool == "magic_wand":
            mask = flood_fill_mask(arr, img_x, img_y, TOLERANCE)
            arr[mask, 3] = 0
        elif self.tool == "color_range":
            mask = color_range_mask(arr, img_x, img_y, TOLERANCE)
            arr[mask, 3] = 0
        elif self.tool == "erase":
            y1 = max(0, img_y - BRUSH_SIZE)
            y2 = min(h, img_y + BRUSH_SIZE)
            x1 = max(0, img_x - BRUSH_SIZE)
            x2 = min(w, img_x + BRUSH_SIZE)
            arr[y1:y2, x1:x2, 3] = 0
        elif self.tool == "restore":
            if self.orig_img is not None:
                orig_arr = np.array(self.orig_img)
                y1 = max(0, img_y - BRUSH_SIZE)
                y2 = min(h, img_y + BRUSH_SIZE)
                x1 = max(0, img_x - BRUSH_SIZE)
                x2 = min(w, img_x + BRUSH_SIZE)
                arr[y1:y2, x1:x2] = orig_arr[y1:y2, x1:x2]

        self.current_img = Image.fromarray(arr)
        self._render()

    def on_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self.zoom = min(self.zoom * ZOOM_STEP, MAX_ZOOM)
        else:
            self.zoom = max(self.zoom / ZOOM_STEP, MIN_ZOOM)
        self._render()

    def undo(self):
        if self.undo_stack:
            self.current_img = self.undo_stack.pop()
            self._render()
            self.status_var.set("Undo applied")

    # ── Actions ───────────────────────────────────────────────
    def preview_image(self):
        if self.current_img:
            self.current_img.show()

    def keep_image(self):
        if self.current_img is None:
            return
        path        = self.queue[self.index]
        folder_name = path.parent.name
        out_path    = get_final_filename(folder_name)
        resized     = resize_to_256(self.current_img)
        resized.save(out_path, "PNG")
        self.status_var.set(f"✅ Saved: {os.path.basename(out_path)}")
        self.index += 1
        self.load_image()

    def skip_image(self):
        self.index += 1
        self.load_image()
        self.status_var.set("⏭ Skipped")

    def remove_background(self):
        if self.current_img is None or self.rembg_running:
            return
        self.pre_rembg = self.current_img.copy()
        self.rembg_running = True
        self.status_var.set("🪄 Removing background...")

        def _run():
            try:
                result = remove(self.current_img)
                self.current_img = result.convert("RGBA")
                self.orig_img    = self.current_img.copy()
                self.root.after(0, self._render)
                self.root.after(0, lambda: self.status_var.set(
                    "✅ Background removed — use Restore brush to fix edges"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(
                    f"❌ rembg error: {e}"))
            finally:
                self.rembg_running = False

        threading.Thread(target=_run, daemon=True).start()


# =============================================================
# SECTION 5 — Entry Point
# =============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app  = ImageReviewApp(root)
    root.mainloop()
