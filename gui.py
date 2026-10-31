"""
gui.py
------
Tkinter GUI for the Polygon Correction Tool.
Imports processing logic from processor.py; no spatial code here.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys

# Ensure the local processor module is importable when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processor import run_correction


# ── Palette ────────────────────────────────────────────────────────────────────
DARK_BG      = "#0F1117"
PANEL_BG     = "#161B25"
CARD_BG      = "#1E2533"
BORDER       = "#2A3347"
ACCENT       = "#4C8BF5"
ACCENT_DARK  = "#3A72D8"
SUCCESS      = "#34D399"
MOVED_COLOR  = "#60A5FA"
WARNING      = "#FBBF24"
DANGER       = "#F87171"
UNRESOLVED   = "#A78BFA"
TEXT_PRIMARY = "#E8EDF5"
TEXT_MUTED   = "#6B7A99"
FONT_MONO    = ("Courier New", 10)


class StatusCard(tk.Frame):
    """A single metric card in the stats panel."""

    def __init__(self, parent, label: str, color: str, **kw):
        super().__init__(parent, bg=CARD_BG, bd=0, **kw)
        self.configure(highlightbackground=color, highlightthickness=1)

        tk.Label(self, text=label, bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(
                     fill="x", padx=14, pady=(12, 0))

        self._var = tk.StringVar(value="0")
        tk.Label(self, textvariable=self._var, bg=CARD_BG, fg=color,
                 font=("Segoe UI", 28, "bold"), anchor="w").pack(
                     fill="x", padx=14, pady=(2, 12))

    def set(self, value):
        self._var.set(str(value))


class FileSelector(tk.Frame):
    """Label + entry + browse button row."""

    def __init__(self, parent, label: str, **kw):
        super().__init__(parent, bg=PANEL_BG, **kw)

        tk.Label(self, text=label, bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        row = tk.Frame(self, bg=PANEL_BG)
        row.pack(fill="x")

        self._path_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=self._path_var,
                         bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                         relief="flat", font=("Segoe UI", 10),
                         highlightbackground=BORDER, highlightthickness=1)
        entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 8))

        btn = tk.Button(row, text="Browse…", command=self._browse,
                        bg=CARD_BG, fg=ACCENT,
                        activebackground=BORDER, activeforeground=ACCENT,
                        relief="flat", font=("Segoe UI", 9, "bold"),
                        cursor="hand2", padx=14, pady=7,
                        highlightbackground=BORDER, highlightthickness=1)
        btn.pack(side="right")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Shapefile",
            filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")]
        )
        if path:
            self._path_var.set(path)

    @property
    def path(self) -> str:
        return self._path_var.get().strip()


class App(tk.Tk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Building Road Overlap Correction Tool")
        self.configure(bg=DARK_BG)
        self.resizable(True, True)
        self.geometry("980x870")
        self.minsize(820, 750)

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ──
        topbar = tk.Frame(self, bg=PANEL_BG, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡ POLYGON CORRECTION TOOL",
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=20, pady=14)

        tk.Label(topbar, text="v2.0 (90° Orthogonal Trim & Smart Move)",
                 bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="right", padx=20)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x")

        # ── Body ──
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=20, pady=20)

        # Left column: inputs & configuration options
        left = tk.Frame(body, bg=PANEL_BG, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self._build_left(left)

        # Right column: live metrics statistics
        right = tk.Frame(body, bg=PANEL_BG, bd=0, width=280,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._build_right(right)

    def _build_left(self, parent):
        pad = {"padx": 20, "pady": 6}

        # Section: Input Files
        tk.Label(parent, text="INPUT LAYERS", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(10, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 6))

        self._road_sel = FileSelector(parent, "Road Shapefile (LineString)")
        self._road_sel.pack(fill="x", **pad)

        self._poly_sel = FileSelector(parent, "Polygon Shapefile")
        self._poly_sel.pack(fill="x", **pad)

        # AOI Layer Selector
        self._aoi_sel = FileSelector(parent, "AOI Shapefile (Optional)")
        self._aoi_sel.pack(fill="x", **pad)

        # Section: Processing Actions Configuration
        tk.Label(parent, text="PROCESSING ACTIONS", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 6))

        action_frame = tk.Frame(parent, bg=PANEL_BG)
        action_frame.pack(fill="x", padx=20, pady=(2, 4))

        self._enable_trim_var = tk.BooleanVar(value=True)
        self._enable_delete_var = tk.BooleanVar(value=True)
        self._enable_move_var = tk.BooleanVar(value=True)

        cb_style = {
            "bg": PANEL_BG,
            "fg": TEXT_PRIMARY,
            "selectcolor": CARD_BG,
            "activebackground": PANEL_BG,
            "activeforeground": TEXT_PRIMARY,
            "font": ("Segoe UI", 9, "bold"),
            "bd": 0,
            "highlightthickness": 0,
            "cursor": "hand2"
        }

        cb_trim = tk.Checkbutton(
            action_frame,
            text="Enable 90° Orthogonal Trimming (Perpendicular to Polygon Edges)",
            variable=self._enable_trim_var,
            **cb_style
        )
        cb_trim.pack(anchor="w", pady=1)

        cb_move = tk.Checkbutton(
            action_frame,
            text="Enable Shifting (Standalone 1st Priority & Patches <= 2.0m)",
            variable=self._enable_move_var,
            **cb_style
        )
        cb_move.pack(anchor="w", pady=1)

        cb_delete = tk.Checkbutton(
            action_frame,
            text="Enable Purging (Delete Small Slivers <3m²)",
            variable=self._enable_delete_var,
            **cb_style
        )
        cb_delete.pack(anchor="w", pady=1)

        # Section: Output
        tk.Label(parent, text="OUTPUT", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(6, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 6))

        out_frame = tk.Frame(parent, bg=PANEL_BG)
        out_frame.pack(fill="x", **pad)

        tk.Label(out_frame, text="Output Shapefile", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        row = tk.Frame(out_frame, bg=PANEL_BG)
        row.pack(fill="x")

        self._out_var = tk.StringVar()
        tk.Entry(row, textvariable=self._out_var,
                 bg=CARD_BG, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                 relief="flat", font=("Segoe UI", 10),
                 highlightbackground=BORDER, highlightthickness=1).pack(
                     side="left", fill="x", expand=True, ipady=7, padx=(0, 8))

        tk.Button(row, text="Save As…", command=self._choose_output,
                  bg=CARD_BG, fg=ACCENT,
                  activebackground=BORDER, activeforeground=ACCENT,
                  relief="flat", font=("Segoe UI", 9, "bold"),
                  cursor="hand2", padx=14, pady=7,
                  highlightbackground=BORDER, highlightthickness=1).pack(side="right")

        # Progress
        tk.Label(parent, text="PROGRESS", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(8, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 6))

        prog_frame = tk.Frame(parent, bg=PANEL_BG)
        prog_frame.pack(fill="x", padx=20, pady=(0, 4))

        self._prog_label = tk.Label(prog_frame, text="Ready", bg=PANEL_BG,
                                    fg=TEXT_MUTED, font=("Segoe UI", 9))
        self._prog_label.pack(anchor="w", pady=(0, 4))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("GIS.Horizontal.TProgressbar",
                        troughcolor=CARD_BG,
                        background=ACCENT,
                        bordercolor=BORDER,
                        lightcolor=ACCENT,
                        darkcolor=ACCENT)

        self._progress = ttk.Progressbar(parent, style="GIS.Horizontal.TProgressbar",
                                         orient="horizontal", mode="determinate")
        self._progress.pack(fill="x", padx=20, pady=(0, 10))

        # Log
        tk.Label(parent, text="LOG", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(0, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(4, 4))

        log_frame = tk.Frame(parent, bg=CARD_BG,
                             highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self._log = tk.Text(log_frame, bg=CARD_BG, fg=TEXT_PRIMARY,
                            font=FONT_MONO, relief="flat",
                            state="disabled", wrap="word",
                            insertbackground=TEXT_PRIMARY, 
                            height=5)
        scroll = ttk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log.pack(fill="both", expand=True, padx=8, pady=8)

        # Run button
        self._run_btn = tk.Button(
            parent, text="▶  RUN CORRECTION",
            command=self._run,
            bg=ACCENT, fg="#FFFFFF",
            activebackground=ACCENT_DARK, activeforeground="#FFFFFF",
            relief="flat", font=("Segoe UI", 11, "bold"),
            cursor="hand2", pady=10,
        )
        self._run_btn.pack(fill="x", padx=20, pady=(0, 12))

    def _build_right(self, parent):
        tk.Label(parent, text="LIVE STATISTICS", bg=PANEL_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(16, 0))
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(4, 14))

        cards_data = [
            ("Total Overlaps Detected", ACCENT,      "total_overlap"),
            ("Corrected (90° Trimmed)", SUCCESS,     "corrected"),
            ("Shifted (Moved <=2m)",    MOVED_COLOR, "moved"),
            ("Deleted (<3 sq m)",       DANGER,      "deleted"),
            ("Unresolved",              UNRESOLVED,  "unresolved"),
        ]

        self._cards = {}
        for label, color, key in cards_data:
            card = StatusCard(parent, label, color)
            card.pack(fill="x", padx=16, pady=(0, 10))
            self._cards[key] = card

        self._summary = tk.Label(parent, text="", bg=PANEL_BG, fg=TEXT_MUTED,
                                 font=("Segoe UI", 8), wraplength=220, justify="left")
        self._summary.pack(anchor="w", padx=16, pady=(4, 10))

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Save corrected shapefile as",
            defaultextension=".shp",
            filetypes=[("Shapefile", "*.shp")]
        )
        if path:
            self._out_var.set(path)

    def _log_write(self, msg: str):
        """Append a line to the log (thread-safe via after())."""
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", msg + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    def _update_stats(self, stats: dict):
        def _do():
            for key, card in self._cards.items():
                card.set(stats.get(key, 0))
        self.after(0, _do)

    def _update_progress(self, current: int, total: int):
        pct = int(current / total * 100) if total else 0
        def _do():
            self._progress["value"] = pct
            self._prog_label.configure(
                text=f"Evaluating building patch {current:,} / {total:,}  ({pct}%)",
                fg=TEXT_PRIMARY
            )
        self.after(0, _do)

    def _set_ui_running(self, running: bool):
        state = "disabled" if running else "normal"
        self._run_btn.configure(
            state=state,
            text="⏳  Processing…" if running else "▶  RUN CORRECTION",
            bg="#2A3347" if running else ACCENT
        )

    def _run(self):
        road_path = self._road_sel.path
        poly_path = self._poly_sel.path
        aoi_path  = self._aoi_sel.path
        out_path  = self._out_var.get().strip()

        if not road_path or not os.path.isfile(road_path):
            messagebox.showerror("Error", "Please select a valid Road shapefile.")
            return
        if not poly_path or not os.path.isfile(poly_path):
            messagebox.showerror("Error", "Please select a valid Polygon shapefile.")
            return
        if not out_path:
            messagebox.showerror("Error", "Please specify an output shapefile path.")
            return

        # AOI Confirmation Prompt
        if not aoi_path:
            user_choice = messagebox.askyesno(
                "AOI Confirmation",
                "AOI is not provided. Do you want to continue without an AOI?"
            )
            if not user_choice:
                return  # Abort run
        elif not os.path.isfile(aoi_path):
            messagebox.showerror("Error", "The specified AOI Shapefile path does not exist.")
            return

        self._set_ui_running(True)
        self._progress["value"] = 0
        self._prog_label.configure(text="Starting…", fg=TEXT_MUTED)
        self._summary.configure(text="")

        for card in self._cards.values():
            card.set(0)

        self._log_write("=" * 48)
        self._log_write(f"Road layer   : {road_path}")
        self._log_write(f"Polygon layer: {poly_path}")
        self._log_write(f"AOI layer    : {aoi_path if aoi_path else '[None - Full Area Processed]'}")
        self._log_write(f"Output       : {out_path}")
        self._log_write(f"Toggles      : Trim={self._enable_trim_var.get()}, Move={self._enable_move_var.get()}, Delete={self._enable_delete_var.get()}")
        self._log_write("=" * 48)
        self._log_write("Loading layers…")

        def worker():
            try:
                stats = run_correction(
                    road_path=road_path,
                    poly_path=poly_path,
                    output_path=out_path,
                    aoi_path=aoi_path if aoi_path else None,
                    enable_trim=self._enable_trim_var.get(),
                    enable_delete=self._enable_delete_var.get(),
                    enable_move=self._enable_move_var.get(),
                    progress_callback=self._update_progress,
                    stats_callback=self._update_stats,
                    log_callback=self._log_write
                )
                self.after(0, lambda s=stats, p=out_path: self._on_done(s, p))
            except Exception as exc:
                err_msg = str(exc)
                import traceback
                traceback.print_exc()
                self.after(0, lambda m=err_msg: self._on_error(m))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, stats: dict, out_path: str):
        self._set_ui_running(False)
        self._progress["value"] = 100
        self._prog_label.configure(text="Complete ✓", fg=SUCCESS)
        self._update_stats(stats)

        self._log_write("")
        self._log_write("✓ Processing complete")
        self._log_write(f"  Total overlaps : {stats.get('total_overlap', 0):,}")
        self._log_write(f"  Corrected(Trim): {stats.get('corrected', 0):,}")
        self._log_write(f"  Shifted (Moved): {stats.get('moved', 0):,}")
        self._log_write(f"  Deleted (<3sqm): {stats.get('deleted', 0):,}")
        self._log_write(f"  Unresolved     : {stats.get('unresolved', 0):,}")
        self._log_write(f"  Output saved   : {out_path}")
        self._log_write("")

        self._summary.configure(
            text=f"Done. {stats.get('corrected', 0)} trimmed · "
                 f"{stats.get('moved', 0)} moved · "
                 f"{stats.get('deleted', 0)} deleted · "
                 f"{stats.get('unresolved', 0)} unresolved."
        )

        messagebox.showinfo("Done", f"Correction complete.\nOutput: {out_path}")

    def _on_error(self, msg: str):
        self._set_ui_running(False)
        self._prog_label.configure(text="Error ✗", fg=DANGER)
        self._log_write(f"\n✗ ERROR: {msg}\n")
        messagebox.showerror("Processing Error", msg)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()