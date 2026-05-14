"""
Spot the Difference — Desktop Application
OOP + Tkinter GUI + OpenCV image processing
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random


# ══════════════════════════════════════════════════════════════════
#  ALTERATION HIERARCHY  (inheritance + polymorphism)
# ══════════════════════════════════════════════════════════════════

class Alteration:
    """Abstract base class for all image alteration types."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Apply the alteration to a region of the image. Must be overridden."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply()")

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name!r})"


class ColourShiftAlteration(Alteration):
    """Shifts the hue of a region in HSV colour space."""

    def __init__(self):
        super().__init__("Colour Shift")
        self._shift = random.randint(35, 85)

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        result = image.copy()
        roi = result[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self._shift) % 180
        result[y:y + h, x:x + w] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result


class BlurAlteration(Alteration):
    """Applies a heavy Gaussian blur to a region."""

    def __init__(self):
        super().__init__("Blur")
        k = random.choice([19, 23, 27])
        self._kernel = (k, k)

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        result = image.copy()
        roi = result[y:y + h, x:x + w]
        result[y:y + h, x:x + w] = cv2.GaussianBlur(roi, self._kernel, 0)
        return result


class BrightnessAlteration(Alteration):
    """Darkens or brightens a rectangular region noticeably."""

    def __init__(self):
        super().__init__("Brightness")
        self._delta = random.choice([-70, -55, 55, 70, 85])

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        result = image.copy()
        roi = result[y:y + h, x:x + w].astype(np.int32)
        result[y:y + h, x:x + w] = np.clip(roi + self._delta, 0, 255).astype(np.uint8)
        return result


class NoiseAlteration(Alteration):
    """Overlays random pixel noise on a region."""

    def __init__(self):
        super().__init__("Noise")
        self._intensity = random.randint(70, 110)

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        result = image.copy()
        roi = result[y:y + h, x:x + w].astype(np.int32)
        noise = np.random.randint(-self._intensity, self._intensity, roi.shape, dtype=np.int32)
        result[y:y + h, x:x + w] = np.clip(roi + noise, 0, 255).astype(np.uint8)
        return result


class SaturationAlteration(Alteration):
    """Desaturates or over-saturates a region in HSV space."""

    def __init__(self):
        super().__init__("Saturation")
        self._factor = random.choice([0.05, 0.1, 2.8, 3.2])

    def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        result = image.copy()
        roi = result[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self._factor, 0, 255)
        result[y:y + h, x:x + w] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result


# ══════════════════════════════════════════════════════════════════
#  DIFFERENCE REGION  (encapsulation)
# ══════════════════════════════════════════════════════════════════

class DifferenceRegion:
    """Represents one programmatically introduced difference in the image."""

    def __init__(self, x: int, y: int, w: int, h: int, alteration: Alteration):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._alteration = alteration
        self._found = False

    # ── read-only properties ──────────────────
    @property
    def x(self) -> int:       return self._x
    @property
    def y(self) -> int:       return self._y
    @property
    def w(self) -> int:       return self._w
    @property
    def h(self) -> int:       return self._h
    @property
    def found(self) -> bool:  return self._found
    @property
    def alteration(self) -> Alteration: return self._alteration

    def centre(self) -> tuple:
        return (self._x + self._w // 2, self._y + self._h // 2)

    def circle_radius(self) -> int:
        return max(self._w, self._h) // 2 + 12

    def mark_found(self):
        self._found = True

    def overlaps(self, other: "DifferenceRegion", margin: int = 25) -> bool:
        """Return True if this region overlaps other (with a safety margin)."""
        return not (
            self._x + self._w + margin < other._x or
            other._x + other._w + margin < self._x or
            self._y + self._h + margin < other._y or
            other._y + other._h + margin < self._y
        )

    def contains_click(self, cx: int, cy: int, tolerance: int = 45) -> bool:
        """Return True if (cx, cy) falls within the region expanded by tolerance."""
        return (
            self._x - tolerance <= cx <= self._x + self._w + tolerance and
            self._y - tolerance <= cy <= self._y + self._h + tolerance
        )

    def __repr__(self):
        return (f"DifferenceRegion({self._alteration.name}, "
                f"x={self._x}, y={self._y}, found={self._found})")


# ══════════════════════════════════════════════════════════════════
#  IMAGE PROCESSOR  (all OpenCV work lives here)
# ══════════════════════════════════════════════════════════════════

class ImageProcessor:
    """Loads images, generates differences, draws markers — pure OpenCV."""

    _ALTERATION_CLASSES = [
        ColourShiftAlteration,
        BlurAlteration,
        BrightnessAlteration,
        NoiseAlteration,
        SaturationAlteration,
    ]

    NUM_DIFFERENCES  = 5
    MIN_REGION       = 45
    MAX_REGION       = 95
    BORDER_MARGIN    = 25

    def __init__(self):
        self._original: np.ndarray | None = None
        self._modified: np.ndarray | None = None
        self._regions: list[DifferenceRegion] = []
        self._img_w: int = 0
        self._img_h: int = 0

    # ── public properties ─────────────────────
    @property
    def original(self) -> np.ndarray | None: return self._original
    @property
    def modified(self) -> np.ndarray | None: return self._modified
    @property
    def regions(self) -> list: return self._regions
    @property
    def img_size(self) -> tuple: return self._img_w, self._img_h

    def load_image(self, path: str, max_w: int = 600, max_h: int = 480) -> bool:
        """Read, resize to fit canvas, clone, introduce differences."""
        raw = cv2.imread(path)
        if raw is None:
            return False

        h, w = raw.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self._original = resized
        self._modified = resized.copy()
        self._img_w, self._img_h = new_w, new_h
        self._regions = []
        self._generate_differences()
        return True

    def _generate_differences(self):
        """Place NUM_DIFFERENCES non-overlapping random alterations."""
        h, w = self._img_h, self._img_w
        m = self.BORDER_MARGIN
        attempts = 0

        while len(self._regions) < self.NUM_DIFFERENCES and attempts < 2000:
            attempts += 1
            rw = random.randint(self.MIN_REGION, self.MAX_REGION)
            rh = random.randint(self.MIN_REGION, self.MAX_REGION)
            rx = random.randint(m, w - rw - m)
            ry = random.randint(m, h - rh - m)

            alteration = random.choice(self._ALTERATION_CLASSES)()
            candidate   = DifferenceRegion(rx, ry, rw, rh, alteration)

            if any(candidate.overlaps(r) for r in self._regions):
                continue

            self._modified = alteration.apply(self._modified, rx, ry, rw, rh)
            self._regions.append(candidate)

    def draw_circle(self, region: DifferenceRegion, colour_bgr: tuple, thickness: int = 3):
        """Draw a circle on both images to mark a found/revealed difference."""
        cx, cy = region.centre()
        r = region.circle_radius()
        cv2.circle(self._original, (cx, cy), r, colour_bgr, thickness)
        cv2.circle(self._modified, (cx, cy), r, colour_bgr, thickness)

    def to_photo_image(self, which: str) -> ImageTk.PhotoImage:
        """Convert an internal cv2 BGR image to a Tkinter-compatible PhotoImage."""
        src = self._original if which == "original" else self._modified
        rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))


# ══════════════════════════════════════════════════════════════════
#  GAME STATE  (pure game logic, no GUI)
# ══════════════════════════════════════════════════════════════════

class GameState:
    """Tracks current round state and cumulative score across images."""

    MAX_MISTAKES    = 3
    CLICK_TOLERANCE = 45

    def __init__(self):
        self._cumulative_score: int = 0
        self._mistakes: int         = 0
        self._game_over: bool       = False
        self._completed: bool       = False
        self._regions: list[DifferenceRegion] = []

    # ── properties ───────────────────────────
    @property
    def cumulative_score(self) -> int:  return self._cumulative_score
    @property
    def mistakes(self) -> int:          return self._mistakes
    @property
    def game_over(self) -> bool:        return self._game_over
    @property
    def completed(self) -> bool:        return self._completed

    def remaining(self) -> int:
        return sum(1 for r in self._regions if not r.found)

    def found_count(self) -> int:
        return sum(1 for r in self._regions if r.found)

    def load_regions(self, regions: list[DifferenceRegion]):
        """Start a new round with a fresh set of regions."""
        self._regions  = regions
        self._mistakes = 0
        self._game_over = False
        self._completed = False

    def process_click(self, img_x: int, img_y: int) -> tuple:
        """
        Evaluate a player click.

        Returns (result_str, region_or_None):
          'found'        – click matched an unfound region
          'mistake'      – missed, still has guesses left
          'max_mistakes' – missed, used all 3 mistakes
          'blocked'      – game is over or already completed
        """
        if self._game_over or self._completed:
            return ("blocked", None)

        for region in self._regions:
            if region.found:
                continue
            if region.contains_click(img_x, img_y, self.CLICK_TOLERANCE):
                region.mark_found()
                self._cumulative_score += 1
                if self.remaining() == 0:
                    self._completed = True
                return ("found", region)

        # No match → mistake
        self._mistakes += 1
        if self._mistakes >= self.MAX_MISTAKES:
            self._game_over = True
            return ("max_mistakes", None)
        return ("mistake", None)

    def reveal_all(self) -> list:
        """Mark game as over and return the list of unfound regions."""
        unfound = [r for r in self._regions if not r.found]
        self._game_over = True
        return unfound


# ══════════════════════════════════════════════════════════════════
#  GAME APP  (Tkinter GUI — orchestrates everything)
# ══════════════════════════════════════════════════════════════════

class GameApp:
    """Main Tkinter window that wires ImageProcessor and GameState together."""

    # ── palette ──────────────────────────────
    BG       = "#1e1e2e"
    PANEL    = "#2a2a3e"
    FG       = "#cdd6f4"
    ACCENT   = "#89b4fa"
    GREEN    = "#a6e3a1"
    RED      = "#f38ba8"
    BTN_BG   = "#313244"
    CANVAS_W = 600
    CANVAS_H = 480

    # OpenCV BGR colours for markers
    _RED_BGR  = (0,   0,   255)
    _BLUE_BGR = (255, 0,   0)

    def __init__(self, root: tk.Tk):
        self._root = root
        self._root.title("🔍 Spot the Difference")
        self._root.configure(bg=self.BG)
        self._root.resizable(False, False)

        self._processor = ImageProcessor()
        self._state     = GameState()

        # Keep PhotoImage refs alive (Tkinter GC quirk)
        self._photo_orig: ImageTk.PhotoImage | None = None
        self._photo_mod:  ImageTk.PhotoImage | None = None

        self._build_ui()

    # ─────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────

    def _build_ui(self):
        # ── Title ──
        tk.Label(self._root, text="🔍 Spot the Difference",
                 font=("Helvetica", 20, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(pady=(12, 4))

        # ── Stats bar ──
        sf = tk.Frame(self._root, bg=self.PANEL, pady=8)
        sf.pack(fill=tk.X, padx=12, pady=(0, 6))

        self._lbl_remaining = self._make_stat(sf, "Remaining: –")
        self._lbl_mistakes  = self._make_stat(sf, "Mistakes: 0 / 3")
        self._lbl_score     = self._make_stat(sf, "Score: 0")
        self._lbl_status    = self._make_stat(sf, "Load an image to begin", fg=self.ACCENT)

        for w in (self._lbl_remaining, self._lbl_mistakes,
                  self._lbl_score, self._lbl_status):
            w.pack(side=tk.LEFT, padx=18)

        # ── Canvas area ──
        cf = tk.Frame(self._root, bg=self.BG)
        cf.pack(padx=12, pady=4)

        cw, ch = self.CANVAS_W, self.CANVAS_H

        # Left column — original (display only)
        left = tk.Frame(cf, bg=self.BG)
        left.grid(row=0, column=0, padx=(0, 8))
        tk.Label(left, text="Original  (reference)",
                 font=("Helvetica", 11, "bold"),
                 bg=self.BG, fg=self.FG).pack(pady=(0, 4))
        self._canvas_orig = tk.Canvas(left, width=cw, height=ch,
                                      bg=self.PANEL,
                                      highlightbackground=self.ACCENT,
                                      highlightthickness=2)
        self._canvas_orig.pack()

        # Right column — modified (clickable)
        right = tk.Frame(cf, bg=self.BG)
        right.grid(row=0, column=1, padx=(8, 0))
        tk.Label(right, text="Modified  ← click here!",
                 font=("Helvetica", 11, "bold"),
                 bg=self.BG, fg=self.FG).pack(pady=(0, 4))
        self._canvas_mod = tk.Canvas(right, width=cw, height=ch,
                                     bg=self.PANEL,
                                     highlightbackground=self.RED,
                                     highlightthickness=2,
                                     cursor="crosshair")
        self._canvas_mod.pack()
        self._canvas_mod.bind("<Button-1>", self._on_click)

        # Placeholder text
        placeholder_cfg = dict(fill=self.FG, font=("Helvetica", 13))
        self._canvas_orig.create_text(cw // 2, ch // 2,
                                      text="Original\n(no image loaded)",
                                      tags="ph", **placeholder_cfg)
        self._canvas_mod.create_text(cw // 2, ch // 2,
                                     text="Modified\n(no image loaded)",
                                     tags="ph", **placeholder_cfg)

        # ── Buttons ──
        bf = tk.Frame(self._root, bg=self.BG)
        bf.pack(pady=12)

        btn_kw = dict(font=("Helvetica", 11, "bold"),
                      bg=self.BTN_BG, fg=self.FG,
                      relief=tk.FLAT, padx=20, pady=8,
                      activebackground=self.ACCENT,
                      activeforeground=self.BG,
                      cursor="hand2", borderwidth=0)

        tk.Button(bf, text="📂  Load Image",
                  command=self._load_image, **btn_kw).pack(side=tk.LEFT, padx=10)

        self._btn_reveal = tk.Button(bf, text="👁  Reveal All",
                                     command=self._reveal_all,
                                     state=tk.DISABLED, **btn_kw)
        self._btn_reveal.pack(side=tk.LEFT, padx=10)

    def _make_stat(self, parent, text, fg=None) -> tk.Label:
        return tk.Label(parent, text=text,
                        font=("Helvetica", 11, "bold"),
                        bg=self.PANEL, fg=fg or self.FG)

    # ─────────────────────────────────────────
    #  Event handlers
    # ─────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return

        ok = self._processor.load_image(path, max_w=self.CANVAS_W, max_h=self.CANVAS_H)
        if not ok:
            messagebox.showerror("Load Error",
                                 "Could not read the image.\nPlease try a different file.")
            return

        self._state.load_regions(self._processor.regions)
        self._redraw_canvases()
        self._update_stats()
        self._set_status("Find the 5 hidden differences!", self.ACCENT)
        self._btn_reveal.config(state=tk.NORMAL)

    def _on_click(self, event: tk.Event):
        if self._processor.original is None:
            return
        if self._state.game_over or self._state.completed:
            return

        ix, iy = self._canvas_to_image(event.x, event.y)
        result, region = self._state.process_click(ix, iy)

        if result == "found":
            self._processor.draw_circle(region, self._RED_BGR)
            self._redraw_canvases()
            self._update_stats()
            rem = self._state.remaining()
            self._set_status(f"✅ Correct!  {rem} difference(s) remaining.", self.GREEN)
            if self._state.completed:
                self._on_all_found()

        elif result == "mistake":
            left = GameState.MAX_MISTAKES - self._state.mistakes
            self._update_stats()
            self._set_status(f"❌ Wrong spot!  {left} guess(es) left.", self.RED)

        elif result == "max_mistakes":
            self._update_stats()
            self._set_status("🚫 Out of guesses!  Load a new image to try again.", self.RED)
            self._btn_reveal.config(state=tk.DISABLED)
            messagebox.showwarning(
                "Game Over — Too Many Mistakes",
                f"You used all 3 guesses!\n\n"
                f"Differences found this round : {self._state.found_count()} / 5\n"
                f"Cumulative score             : {self._state.cumulative_score}\n\n"
                f"Load a new image to keep playing."
            )

    def _reveal_all(self):
        if self._processor.original is None:
            return
        unfound = self._state.reveal_all()
        for region in unfound:
            self._processor.draw_circle(region, self._BLUE_BGR)
        self._redraw_canvases()
        self._update_stats()
        self._set_status("🔵 All differences revealed.  Load a new image.", self.ACCENT)
        self._btn_reveal.config(state=tk.DISABLED)
        messagebox.showinfo(
            "Differences Revealed",
            f"Revealed {len(unfound)} unfound difference(s).\n"
            f"Cumulative score : {self._state.cumulative_score}\n\n"
            f"Load a new image to continue!"
        )

    def _on_all_found(self):
        self._btn_reveal.config(state=tk.DISABLED)
        self._set_status("🎉 All 5 found!  Load a new image to keep going!", self.GREEN)
        messagebox.showinfo(
            "Congratulations! 🎉",
            f"You found all 5 differences!\n\n"
            f"Mistakes this round  : {self._state.mistakes}\n"
            f"Cumulative score     : {self._state.cumulative_score}\n\n"
            f"Load another image to continue!"
        )

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────

    def _canvas_to_image(self, cx: int, cy: int) -> tuple:
        """Convert canvas pixel coords to image pixel coords (account for centering)."""
        iw, ih = self._processor.img_size
        ox = (self.CANVAS_W - iw) // 2
        oy = (self.CANVAS_H - ih) // 2
        return cx - ox, cy - oy

    def _redraw_canvases(self):
        """Push updated OpenCV images into both Tkinter canvases."""
        self._photo_orig = self._processor.to_photo_image("original")
        self._photo_mod  = self._processor.to_photo_image("modified")

        iw, ih = self._processor.img_size
        ox = (self.CANVAS_W - iw) // 2
        oy = (self.CANVAS_H - ih) // 2

        for canvas, photo, tag in [
            (self._canvas_orig, self._photo_orig, "img_orig"),
            (self._canvas_mod,  self._photo_mod,  "img_mod"),
        ]:
            canvas.delete("ph")
            canvas.delete(tag)
            canvas.create_image(ox, oy, anchor=tk.NW, image=photo, tags=tag)

    def _update_stats(self):
        self._lbl_remaining.config(
            text=f"Remaining: {self._state.remaining()}")
        self._lbl_mistakes.config(
            text=f"Mistakes: {self._state.mistakes} / {GameState.MAX_MISTAKES}",
            fg=self.RED if self._state.mistakes > 0 else self.FG)
        self._lbl_score.config(
            text=f"Score: {self._state.cumulative_score}")

    def _set_status(self, msg: str, colour: str = None):
        self._lbl_status.config(text=msg, fg=colour or self.FG)


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    GameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
