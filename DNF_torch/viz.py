import pygame
import pygame_gui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import time
import os
import json
import matplotlib.cm as cm

# Your Field class (safe import for both package/script contexts)
try:
    from .field import Field  # when part of a package
except Exception:
    from field import Field   # when run as a script


# ----------------------------
# Simple Matplotlib animation
# ----------------------------
def animate_fields(fields, steps=100, interval=100):
    """
    Quickly animate one or more Field activations using Matplotlib.
    """
    fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields), 4))
    if len(fields) == 1:
        axes = [axes]

    ims = []
    for ax, f in zip(axes, fields):
        im = ax.imshow(
            f.get_activation().squeeze().cpu().numpy(),
            cmap='viridis', origin='lower', animated=True
        )
        ax.set_title("Field Step 0")
        ims.append(im)

    def update(step):
        for f, im in zip(fields, ims):
            f.update()
            im.set_array(f.get_activation().squeeze().cpu().numpy())
            im.axes.set_title(f"Field Step {step+1}")
        return ims

    animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
    plt.show()


# ----------------------------
# Pygame tuner GUI
# ----------------------------
class FieldTuner:
    """
    Pygame GUI to pick a field, tune params in real time, render activation,
    and save current parameters to JSON.
    """
    def __init__(self, fields):
        pygame.init()
        self.W, self.H = 1000, 720
        pygame.display.set_caption(f"System with {len(fields)} fields")
        self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
        self.manager = pygame_gui.UIManager((self.W, self.H))
        self.clock = pygame.time.Clock()

        # custom event that will NOT collide with pygame_gui (avoid AttributeError)
        self.SAVE_RESET_EVENT = pygame.event.custom_type()

        self.fields = fields
        self.current_field = None

        # --- layout ---
        self.btn_x, self.btn_y0 = 20, 60
        self.btn_w, self.btn_h, self.btn_gap = 210, 40, 12

        self.viz_rect = pygame.Rect(260, 20, 720, 500)
        self.sliders_area = pygame.Rect(260, 540, 720, 160)

        self.title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(20, 20, 220, 30),
            text=f"System with {len(fields)} fields",
            manager=self.manager
        )

        # Field selection buttons
        self.buttons = []
        y = self.btn_y0
        for i, f in enumerate(fields):
            shape = getattr(f, "HW", None)
            shape_str = f"{shape[0]}x{shape[1]}" if shape and len(shape) == 2 else "?"
            b = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(self.btn_x, y, self.btn_w, self.btn_h),
                text=f"Field {i+1} ({shape_str})",
                manager=self.manager,
            )
            self.buttons.append(b)
            y += self.btn_h + self.btn_gap

        # SAVE button
        self.save_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(self.btn_x, y + 10, self.btn_w, self.btn_h),
            text="SAVE",
            manager=self.manager,
        )

        # sliders bookkeeping
        self.sliders = {}
        self.slider_labels = {}
        self.slider_value_labels = {}
        self.slider_specs = {}  # name -> dict(spec)

        # viz background tile
        self.viz_bg = pygame.Surface((self.viz_rect.w, self.viz_rect.h))
        self.viz_bg.fill((18, 18, 18))

    # ---------------- slider helpers ----------------
    def _clear_sliders(self):
        for w in list(self.sliders.values()) + list(self.slider_labels.values()) + list(self.slider_value_labels.values()):
            w.kill()
        self.sliders.clear()
        self.slider_labels.clear()
        self.slider_value_labels.clear()
        self.slider_specs.clear()

    def _create_sliders(self, field):
        """
        Build sliders based on your spec:
          (name, default_value, (min, max), sign_str, is_log)
        For log sliders, default and (min,max) are in log10 space; UI shows actual 10**value.
        """
        spec_list = [
            ("global_inhibition", math.log10(0.010), (-2.0, 1.1),  "+", True),
            ("beta",              math.log10(3.388), (0.0, 1.3),   "+", True),
            ("noise_strength",    math.log10(0.012), (-2.0, -0.3), "+", True),
            ("resting_level",     -0.0,              (-20.0, 0.0), "",  False),
            ("scale",             math.log10(7.133), (-2.0, 1.0),  "+", True),
            ("kernel_scale",      math.log10(0.272), (-2.0, 1.0),  "+", True),
            ("time_scale",        59.40,             (1.0, 200.0), "+", False),
            ("ior_scale",         0.65,              (0.0, 1.0),   "+", False),
        ]

        x0, y0 = self.sliders_area.x + 8, self.sliders_area.y + 6
        col_w = 340
        row_h = 50      # slightly taller to avoid label height warnings
        per_col = 4     # sliders per column

        for idx, (name, default_val, (vmin, vmax), sign, is_log) in enumerate(spec_list):
            # Determine start value: if field has attr, use it; otherwise use default.
            if hasattr(field, name):
                actual = getattr(field, name)
                start = math.log10(actual) if is_log and actual > 0 else (default_val if is_log else actual)
            else:
                actual = (10 ** default_val) if is_log else default_val
                start = default_val if is_log else default_val
                setattr(field, name, actual)

            # Column layout
            col = idx // per_col
            row = idx % per_col
            px = x0 + col * col_w
            py = y0 + row * row_h

            # Label (wider/taller to avoid "size diff" warnings)
            lbl = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(px, py, 220, 22),
                text=f"{name}" + (" (log10)" if is_log else ""),
                manager=self.manager,
            )
            # Slider
            sld = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(px, py + 22, 240, 24),
                start_value=float(start),
                value_range=(float(vmin), float(vmax)),
                manager=self.manager,
            )
            # Value label
            val_lbl = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(px + 245, py + 22, 110, 24),
                text=self._format_value(name, start, is_log),
                manager=self.manager,
            )

            self.slider_labels[name] = lbl
            self.sliders[name] = sld
            self.slider_value_labels[name] = val_lbl
            self.slider_specs[name] = {"is_log": is_log, "vmin": vmin, "vmax": vmax}

    def _format_value(self, name, slider_value, is_log):
        if is_log:
            actual = 10 ** float(slider_value)
            return f"{actual:.3g}"
        else:
            return f"{float(slider_value):.3g}"

    def _get_all_slider_values(self):
        """
        Returns a dict of {param_name: actual_value} for the current field.
        """
        values = {}
        for name, slider in self.sliders.items():
            v = float(slider.get_current_value())
            if self.slider_specs[name]["is_log"]:
                values[name] = 10 ** v
            else:
                values[name] = v
        return values

    # ---------------- visualization helpers ----------------
    @staticmethod
    def _normalize_to_uint8(a: np.ndarray) -> np.ndarray:
        # keep for 1-D line plots
        a = np.asarray(a, dtype=np.float32)
        amin = np.nanmin(a) if a.size else 0.0
        amax = np.nanmax(a) if a.size else 1.0
        if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
            return np.zeros_like(a, dtype=np.uint8)
        return ((a - amin) / (amax - amin) * 255).clip(0, 255).astype(np.uint8)

    def _draw_1d(self, arr: np.ndarray, surface: pygame.Surface):
        surface.fill((12, 12, 12))
        h, w = surface.get_height(), surface.get_width()
        if arr.size < 2:
            return
        y = self._normalize_to_uint8(arr).astype(np.float32)
        y_scaled = (h - 10) * (1.0 - (y / 255.0)) + 5
        xs = np.linspace(5, w - 5, num=y_scaled.size).astype(int)
        pts = list(zip(xs.tolist(), y_scaled.astype(int).tolist()))
        pygame.draw.lines(surface, (80, 200, 255), False, pts, 2)

    def _make_surface_from_2d(self, arr2d: np.ndarray) -> pygame.Surface:
        """
        Map 2D array -> RGB surface using viridis colormap.
        """
        a = np.asarray(arr2d, dtype=np.float32)
        amin = np.nanmin(a) if a.size else 0.0
        amax = np.nanmax(a) if a.size else 1.0
        if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
            a = np.zeros_like(a, dtype=np.float32)
            amin, amax = 0.0, 1.0
        normed = (a - amin) / (amax - amin)
        rgb = cm.viridis(normed)[..., :3]      # (H, W, 3) floats 0..1
        rgb8 = (rgb * 255).astype(np.uint8)
        # pygame wants (W, H, 3) transposed
        surf = pygame.surfarray.make_surface(np.transpose(rgb8, (1, 0, 2)))
        return surf

    def _draw_2d(self, arr2d: np.ndarray, surface: pygame.Surface):
        surface.blit(self.viz_bg, (0, 0))
        surf = self._make_surface_from_2d(arr2d)
        sw, sh = surf.get_width(), surf.get_height()
        tw, th = surface.get_width(), surface.get_height()
        scale = min(tw / sw, th / sh)
        new_size = (max(1, int(sw * scale)), max(1, int(sh * scale)))
        surf = pygame.transform.smoothscale(surf, new_size)
        x = (tw - new_size[0]) // 2
        y = (th - new_size[1]) // 2
        surface.blit(surf, (x, y))

    def _draw_3d_slices(self, arr3d: np.ndarray, surface: pygame.Surface, max_slices: int = 6):
        surface.blit(self.viz_bg, (0, 0))
        D = arr3d.shape[0]
        idxs = np.linspace(0, D - 1, num=min(D, max_slices)).astype(int)
        gap = 8
        tile_w = (surface.get_width() - gap * (len(idxs) + 1)) // len(idxs)
        tile_h = surface.get_height() - 2 * gap
        x = gap
        y = gap
        for i in idxs:
            surf = self._make_surface_from_2d(arr3d[i])
            surf = pygame.transform.smoothscale(surf, (tile_w, tile_h))
            surface.blit(surf, (x, y))
            x += tile_w + gap

    def _render_current_field(self):
        if self.current_field is None:
            pygame.draw.rect(self.screen, (40, 40, 40), self.viz_rect, border_radius=6)
            return
        act = self.current_field.get_activation().detach().cpu().squeeze().numpy()
        viz_surface = pygame.Surface((self.viz_rect.w, self.viz_rect.h))
        if act.ndim == 1:
            self._draw_1d(act, viz_surface)
        elif act.ndim == 2:
            self._draw_2d(act, viz_surface)
        elif act.ndim == 3:
            self._draw_3d_slices(act, viz_surface, max_slices=6)
        else:
            viz_surface.fill((25, 25, 25))
        self.screen.blit(viz_surface, self.viz_rect.topleft)

    # ---------------- saving ----------------
    def _ensure_params_dir(self):
        params_dir = os.path.join(os.getcwd(), "field_params")
        os.makedirs(params_dir, exist_ok=True)
        return params_dir

    def _field_identifier(self, f, idx):
        # Prefer a user-defined name; else shape; else index
        name = getattr(f, "name", None)
        if name:
            return str(name)
        shape = getattr(f, "HW", None)
        if shape and isinstance(shape, (tuple, list)) and len(shape) == 2:
            return f"field_{idx+1}_{shape[0]}x{shape[1]}"
        return f"field_{idx+1}"

    def _save_current_params(self):
        # reflect current sliders into the selected field before dumping
        if self.current_field is not None and self.sliders:
            for k, v in self._get_all_slider_values().items():
                setattr(self.current_field, k, v)

        # collect params for ALL fields
        all_fields_payload = []
        # Use slider_specs as the canonical param list (created after a field is opened)
        param_names = list(self.slider_specs.keys())
        for i, f in enumerate(self.fields):
            per_field = {}
            for name in param_names:
                if hasattr(f, name):
                    per_field[name] = getattr(f, name)
            all_fields_payload.append({
                "id": self._field_identifier(f, i),
                "index": i,
                "params": per_field
            })

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
            "fields": all_fields_payload,
        }

        params_dir = self._ensure_params_dir()
        fname = os.path.join(params_dir, f"fields_{payload['timestamp']}.json")
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)

        self.save_button.set_text("SAVED ✓")
        # schedule reset via our custom event (no collision with pygame_gui)
        pygame.time.set_timer(self.SAVE_RESET_EVENT, 800, loops=1)

    # ---------------- main loop ----------------
    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(30) / 1000.0
            for event in pygame.event.get():

                # our timer event: handle it and DO NOT pass to pygame_gui
                if event.type == self.SAVE_RESET_EVENT:
                    self.save_button.set_text("SAVE")
                    continue

                # window resizing: keep UI manager informed and keep surface resizable
                if event.type == pygame.VIDEORESIZE:
                    self.W, self.H = event.w, event.h
                    self.screen = pygame.display.set_mode((self.W, self.H), pygame.RESIZABLE)
                    self.manager.set_window_resolution((self.W, self.H))
                    # (optional) reflow draw-only rects:
                    right_w = max(320, self.W - 260 - 20)  # 260 left panel + 20 margin
                    top_h = max(300, int(self.H * 0.68))
                    bottom_h = max(120, self.H - (20 + top_h + 20))
                    self.viz_rect = pygame.Rect(260, 20, right_w, top_h)
                    self.sliders_area = pygame.Rect(260, 20 + top_h + 20, right_w, bottom_h)
                    self.viz_bg = pygame.Surface((self.viz_rect.w, self.viz_rect.h))
                    self.viz_bg.fill((18, 18, 18))
                    # continue; we still want pygame_gui to get the resize

                if event.type == pygame.QUIT:
                    running = False
                    continue

                # Let pygame_gui process remaining events
                self.manager.process_events(event)

                # UI events
                if event.type == pygame.USEREVENT and hasattr(event, "user_type"):
                    if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.save_button:
                            self._save_current_params()
                        else:
                            for idx, b in enumerate(self.buttons):
                                if event.ui_element == b:
                                    self.current_field = self.fields[idx]
                                    self._clear_sliders()
                                    self._create_sliders(self.current_field)

                    elif event.user_type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        if self.current_field is not None:
                            for name, slider in self.sliders.items():
                                if event.ui_element == slider:
                                    is_log = self.slider_specs[name]["is_log"]
                                    sval = float(slider.get_current_value())
                                    actual = 10 ** sval if is_log else sval
                                    setattr(self.current_field, name, actual)
                                    self.slider_value_labels[name].set_text(f"{actual:.3g}")

            # step all fields
            for f in self.fields:
                f.forward()

            # update UI
            self.manager.update(time_delta)

            # draw (viz first so UI stays on top)
            self.screen.fill((30, 30, 30))
            self._render_current_field()
            self.manager.draw_ui(self.screen)
            pygame.display.flip()

        pygame.quit()


def run_tuner(fields):
    FieldTuner(fields).run()
