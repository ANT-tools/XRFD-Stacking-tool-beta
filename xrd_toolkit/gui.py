from __future__ import annotations

import csv
import json
from pathlib import Path
import queue
import threading
import traceback

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .io_utils import (
    load_tiff,
    list_tiffs,
    inspect_tiff_output_shape,
    inspect_tiff_exposure_seconds,
    inspect_tiff_exposure_info,
    select_exposure_from_info,
    save_float_tiff,
    save_mask_tiff,
    save_csv,
    save_json,
)
from .processing import (
    ProcessingSettings, ProcessingEngine, display_level_bounds,
    parse_custom_curve_spec,
)
from .analysis import extract_meridian_equator, find_profile_peaks
from .stacking import StackSettings, build_stack, preflight_stack_shapes
from .symmetry import SymmetrySettings, build_symmetry_average
from .provenance import utc_timestamp, sha256_file, write_provenance_bundle
from .backend import get_gpu_info, resolve_backend
from .axis_detection import AxisDetectionSettings, detect_beam_center_and_fiber_axis
from .layer_lines import (
    detector_to_fiber, fiber_to_detector, fit_evenly_spaced_layer_lines,
    suggest_anchor_from_axial_profile, sample_fiber_strip, fit_profile_peaks,
)
from .roi import ROIShape, measure_roi
from .comparison import compare_images
from .session import save_session, load_session


APP_TITLE = "XRD Image Toolkit 0.7.2"


class XRDToolkitApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1750x930")
        self.minsize(1350, 760)

        # ---------------- Data state ----------------
        self.current_path = None
        self.current_label = None
        self.current_source_kind = None

        self.raw = None
        self.dark = None
        self.flat = None
        self.background = None

        self.dark_path = None
        self.flat_path = None
        self.background_path = None

        self.results = None
        self.processing_engine = ProcessingEngine()

        self.folder_path = None
        self.folder_entries = []
        self.folder_common_shape = None
        self._tree_selection_anchor = None

        self.stack_result = None
        self.symmetry_result = None
        self.axis_detection_result = None
        self.use_symmetry_for_profiles = tk.BooleanVar(value=False)

        self.click_mode = None
        self.axis_click_points = []

        # Interactive analysis state
        self.rois = []
        self.roi_counter = 0
        self.roi_draw_mode = None
        self.roi_drag_start = None
        self.roi_preview_patch = None
        self.layer_line_result = None
        self.layer_line_peak_results = {}
        self.comparison_a = None
        self.comparison_b = None
        self.comparison_a_label = None
        self.comparison_b_label = None
        self.comparison_result = None

        self.worker_queue = queue.Queue()
        self.worker_active = False
        self.axis_worker_queue = queue.Queue()
        self.axis_worker_active = False

        # ---------------- Tk variables ----------------
        self.view_var = tk.StringVar(value="Display")
        self.cmap_var = tk.StringVar(value="gray")

        # Corrections
        self.dark_enabled = tk.BooleanVar(value=False)
        self.flat_enabled = tk.BooleanVar(value=False)

        self.background_enabled = tk.BooleanVar(value=False)
        self.background_scale = tk.DoubleVar(value=1.0)

        self.normalize_enabled = tk.BooleanVar(value=False)
        self.monitor_value = tk.DoubleVar(value=1.0)

        self.hot_pixels_enabled = tk.BooleanVar(value=False)
        self.hot_pixel_sigma = tk.DoubleVar(value=8.0)
        self.hot_pixel_size = tk.IntVar(value=3)

        self.saturation_enabled = tk.BooleanVar(value=False)
        self.saturation_value = tk.DoubleVar(value=65535.0)

        self.beamstop_enabled = tk.BooleanVar(value=False)
        self.beamstop_center_x = tk.DoubleVar(value=0.0)
        self.beamstop_center_y = tk.DoubleVar(value=0.0)
        self.beamstop_radius = tk.DoubleVar(value=0.0)

        # Display / filters
        self.median_filter_enabled = tk.BooleanVar(value=False)
        self.median_filter_size = tk.IntVar(value=3)

        self.gaussian_filter_enabled = tk.BooleanVar(value=False)
        self.gaussian_filter_sigma = tk.DoubleVar(value=1.0)

        self.gaussian_background_enabled = tk.BooleanVar(value=False)
        self.gaussian_background_sigma = tk.DoubleVar(value=40.0)

        self.median_background_enabled = tk.BooleanVar(value=False)
        self.median_background_size = tk.IntVar(value=41)

        self.high_pass_enabled = tk.BooleanVar(value=False)
        self.high_pass_sigma = tk.DoubleVar(value=6.0)

        self.unsharp_enabled = tk.BooleanVar(value=False)
        self.unsharp_sigma = tk.DoubleVar(value=2.0)
        self.unsharp_amount = tk.DoubleVar(value=1.0)

        self.display_mode = tk.StringVar(value="log")
        self.log_gain = tk.DoubleVar(value=100.0)
        self.gamma = tk.DoubleVar(value=0.5)
        self.asinh_strength = tk.DoubleVar(value=20.0)
        self.custom_curve_spec = tk.StringVar(value="0,0;1,1")
        self.tone_curve_monotonic = tk.BooleanVar(value=True)
        self.tone_curve_points = [(0.0, 0.0), (1.0, 1.0)]
        self.tone_curve_drag_index = None
        self.tone_curve_selected_index = None
        self.contrast_mode = tk.StringVar(value="legacy percentile")
        self.percentile_low = tk.DoubleVar(value=0.5)
        self.percentile_high = tk.DoubleVar(value=99.7)
        self.manual_black = tk.DoubleVar(value=0.0)
        self.manual_white = tk.DoubleVar(value=1.0)
        self.robust_sigma = tk.DoubleVar(value=6.0)
        self.invert_display = tk.BooleanVar(value=False)
        self.local_contrast_enabled = tk.BooleanVar(value=False)
        self.local_contrast_sigma = tk.DoubleVar(value=25.0)
        self.local_contrast_strength = tk.DoubleVar(value=0.45)
        self.local_contrast_noise_floor = tk.DoubleVar(value=0.15)
        self.histogram_bins = tk.IntVar(value=2048)

        # Fiber analysis
        self.center_x = tk.DoubleVar(value=0.0)
        self.center_y = tk.DoubleVar(value=0.0)
        self.fiber_angle = tk.DoubleVar(value=90.0)
        self.strip_width = tk.IntVar(value=7)
        self.peak_prominence = tk.DoubleVar(value=0.0)
        self.peak_distance = tk.DoubleVar(value=0.0)
        self.axis_analysis_max_dimension = tk.IntVar(value=700)
        self.axis_center_search_radius = tk.DoubleVar(value=30.0)
        self.axis_center_refine_radius = tk.DoubleVar(value=3.0)
        self.axis_center_refine_step = tk.DoubleVar(value=0.5)
        self.axis_coarse_angle_step = tk.DoubleVar(value=2.0)
        self.axis_fine_angle_step = tk.DoubleVar(value=0.1)
        self.axis_central_exclusion = tk.DoubleVar(value=20.0)
        self.axis_detection_summary_var = tk.StringVar(value="No automatic detection has been run.")

        # Stacking
        self.stack_method = tk.StringVar(value="mean")
        self.stack_align = tk.BooleanVar(value=False)
        self.stack_max_shift = tk.IntVar(value=12)
        self.stack_sigma = tk.DoubleVar(value=4.0)
        self.stack_iterations = tk.IntVar(value=2)
        self.stack_trim_fraction = tk.DoubleVar(value=0.10)
        self.stack_winsor_fraction = tk.DoubleVar(value=0.05)
        self.stack_huber_delta = tk.DoubleVar(value=1.5)
        self.stack_huber_iterations = tk.IntVar(value=3)
        self.stack_noise_weight_floor = tk.DoubleVar(value=1e-6)
        self.stack_chunk_rows = tk.IntVar(value=128)
        self.stack_normalization_mode = tk.StringVar(value="none")
        self.stack_reference_exposure = tk.DoubleVar(value=1.0)
        self.input_correction_state = tk.StringVar(value="already dark + flat corrected")
        self.exposure_batch_value = tk.DoubleVar(value=1.0)
        self.exposure_batch_unit = tk.StringVar(value="s")
        self.exposure_auto_policy = tk.StringVar(value="filename only")

        # Layer-line analysis
        self.layer_line_count = tk.IntVar(value=7)
        self.layer_line_anchor_offset = tk.DoubleVar(value=0.0)
        self.layer_line_refine_radius = tk.DoubleVar(value=8.0)
        self.layer_line_inner_perp = tk.DoubleVar(value=20.0)
        self.layer_line_outer_perp = tk.DoubleVar(value=0.0)
        self.layer_line_half_width = tk.DoubleVar(value=2.0)
        self.layer_line_selected_index = tk.IntVar(value=0)
        self.layer_line_max_peaks = tk.IntVar(value=4)
        self.layer_line_peak_prominence = tk.DoubleVar(value=0.0)
        self.layer_line_summary_var = tk.StringVar(value="No layer-line model fitted.")

        # ROI / annotation tools
        self.roi_source = tk.StringVar(value="Corrected")
        self.roi_summary_var = tk.StringVar(value="No ROI selected.")

        # Image comparison
        self.comparison_mode = tk.StringVar(value="difference")
        self.comparison_scale_mode = tk.StringVar(value="least-squares fit")
        self.comparison_manual_scale = tk.DoubleVar(value=1.0)
        self.comparison_alpha = tk.DoubleVar(value=0.5)
        self.comparison_exposure_a = tk.DoubleVar(value=1.0)
        self.comparison_exposure_b = tk.DoubleVar(value=1.0)
        self.comparison_summary_var = tk.StringVar(value="Load comparison images A and B.")

        # Symmetry / quadrant-folding settings
        self.symmetry_mode = tk.StringVar(value="four-quadrant")
        self.symmetry_statistic = tk.StringVar(value="mean")
        self.symmetry_half_width = tk.DoubleVar(value=0.0)
        self.symmetry_half_height = tk.DoubleVar(value=0.0)
        self.symmetry_min_contributors = tk.IntVar(value=1)
        self.symmetry_summary_var = tk.StringVar(
            value="No symmetry average has been built."
        )

        # Performance
        self.compute_backend = tk.StringVar(value="auto")
        self.viewer_max_dimension = tk.IntVar(value=1400)
        self.fast_percentiles = tk.BooleanVar(value=True)
        self.percentile_sample_target = tk.IntVar(value=262144)
        self.fast_gaussian_background = tk.BooleanVar(value=True)
        self.gaussian_background_downsample = tk.IntVar(value=4)
        self.fast_median_background = tk.BooleanVar(value=True)
        self.median_background_downsample = tk.IntVar(value=4)
        self.registration_crop_size = tk.IntVar(value=1536)
        self.fft_workers = tk.IntVar(value=-1)
        self.tiff_workers = tk.IntVar(value=0)
        self.gpu_status_var = tk.StringVar(value="Checking GPU…")

        # PNG sharing/export options
        self.png_include_axes = tk.BooleanVar(value=True)
        self.png_include_title = tk.BooleanVar(value=True)
        self.png_dpi = tk.IntVar(value=180)

        # Provenance / detector-coordinate export metadata
        self.provenance_auto_sidecars = tk.BooleanVar(value=True)
        self.provenance_calibration_label = tk.StringVar(value="MD-033")
        self.provenance_expected_rows = tk.IntVar(value=3072)
        self.provenance_expected_columns = tk.IntVar(value=3072)

        self.status_var = tk.StringVar(value="Ready.")
        self.stack_summary_var = tk.StringVar(value="No stack has been built.")

        self._build_gui()
        self._bind_variable_updates()

    # ============================================================
    # GUI BUILD
    # ============================================================

    def _build_gui(self):
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # Permanent folder/file side panel
        file_pane = ttk.Frame(main, width=390)
        controls_pane = ttk.Frame(main, width=410)
        viewer_pane = ttk.Frame(main)

        main.add(file_pane, weight=0)
        main.add(controls_pane, weight=0)
        main.add(viewer_pane, weight=1)

        self._build_file_pane(file_pane)
        self._build_controls(controls_pane)
        self._build_viewer(viewer_pane)

        status = ttk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_file_pane(self, parent):
        title = ttk.Label(
            parent,
            text="TIFF Folder / Stack Selection",
            font=("", 11, "bold"),
        )
        title.pack(fill=tk.X, padx=8, pady=(8, 4))

        buttons = ttk.Frame(parent)
        buttons.pack(fill=tk.X, padx=8, pady=4)

        ttk.Button(
            buttons,
            text="Open TIFF…",
            command=self.load_single_tiff,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))

        ttk.Button(
            buttons,
            text="Open Folder…",
            command=self.load_folder,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.folder_label = ttk.Label(
            parent,
            text="No folder loaded",
            wraplength=370,
            justify=tk.LEFT,
        )
        self.folder_label.pack(fill=tk.X, padx=8, pady=4)

        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        columns = ("use", "file", "shape", "exposure", "exp_source", "header_exp", "exp_qc", "mean", "max", "corr")
        self.file_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
        )

        self.file_tree.heading("use", text="Use")
        self.file_tree.heading("file", text="File")
        self.file_tree.heading("shape", text="Size")
        self.file_tree.heading("exposure", text="Exp(s)")
        self.file_tree.heading("exp_source", text="Source")
        self.file_tree.heading("header_exp", text="Hdr(s)")
        self.file_tree.heading("exp_qc", text="Exp QC")
        self.file_tree.heading("mean", text="Mean")
        self.file_tree.heading("max", text="Max")
        self.file_tree.heading("corr", text="Corr")

        self.file_tree.column("use", width=45, minwidth=40, anchor="center")
        self.file_tree.column("file", width=185, minwidth=110)
        self.file_tree.column("shape", width=95, minwidth=80, anchor="center")
        self.file_tree.column("exposure", width=72, minwidth=60, anchor="e")
        self.file_tree.column("exp_source", width=92, minwidth=75, anchor="w")
        self.file_tree.column("header_exp", width=65, minwidth=58, anchor="e")
        self.file_tree.column("exp_qc", width=90, minwidth=70, anchor="center")
        self.file_tree.column("mean", width=70, minwidth=55, anchor="e")
        self.file_tree.column("max", width=75, minwidth=60, anchor="e")
        self.file_tree.column("corr", width=65, minwidth=55, anchor="e")

        yscroll = ttk.Scrollbar(
            tree_frame,
            orient=tk.VERTICAL,
            command=self.file_tree.yview,
        )
        xscroll = ttk.Scrollbar(
            tree_frame,
            orient=tk.HORIZONTAL,
            command=self.file_tree.xview,
        )
        self.file_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.file_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        self.file_tree.tag_configure("excluded", foreground="#777777")
        self.file_tree.tag_configure("shape_mismatch", foreground="#b00020")
        self.file_tree.tag_configure("exposure_mismatch", background="#fff0b3")
        self.file_tree.tag_configure("exposure_unknown", background="#ffd9d9")

        self.file_tree.bind(
            "<<TreeviewSelect>>",
            self._preview_focused_tree_item,
        )
        # Tk/ttk multi-selection semantics differ subtly by platform. These
        # explicit bindings guarantee additive Ctrl+Shift range selection.
        self.file_tree.bind("<Button-1>", self._tree_plain_click, add="+")
        self.file_tree.bind("<Shift-Button-1>", self._tree_shift_click)
        self.file_tree.bind("<Control-Button-1>", self._tree_ctrl_click)
        self.file_tree.bind("<Control-Shift-Button-1>", self._tree_ctrl_shift_click)
        self.file_tree.bind(
            "<Double-1>",
            self._tree_double_click,
        )
        self.file_tree.bind(
            "<space>",
            self._toggle_highlighted_inclusion,
        )

        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X, padx=8, pady=(4, 2))

        ttk.Button(
            row1,
            text="Include selected",
            command=self.include_selected,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        ttk.Button(
            row1,
            text="Exclude selected",
            command=self.exclude_selected,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Button(
            row1,
            text="Invert",
            command=self.invert_inclusion,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, padx=8, pady=2)

        ttk.Button(
            row2,
            text="Include all",
            command=self.include_all,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        ttk.Button(
            row2,
            text="Exclude all",
            command=self.exclude_all,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Button(
            row2,
            text="Use highlighted only",
            command=self.use_highlighted_only,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        self.include_count_label = ttk.Label(
            parent,
            text="0 / 0 included",
        )
        self.include_count_label.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(
            parent,
            text=(
                "Click selects one row; Ctrl+click toggles rows; Shift+click selects a range; "
                "Ctrl+Shift+click adds a range to the existing selection. Double-click, press "
                "Space, or use the buttons above to "
                "change whether selected frames are included in the stack."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=8, pady=(2, 8))

    def _create_scrollable_notebook_tab(self, text):
        """Create a notebook page with a vertically scrollable content frame.

        The returned content frame can be used exactly like the previous static
        tab frame, so existing tab builders do not need to know about the canvas.
        """
        page = ttk.Frame(self.notebook)
        self.notebook.add(page, text=text)

        canvas = tk.Canvas(
            page,
            highlightthickness=0,
            borderwidth=0,
            takefocus=False,
        )
        scrollbar = ttk.Scrollbar(
            page,
            orient=tk.VERTICAL,
            command=canvas.yview,
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        content = ttk.Frame(canvas)
        window_id = canvas.create_window(
            (0, 0),
            window=content,
            anchor="nw",
        )

        def update_scroll_region(_event=None):
            bbox = canvas.bbox("all")
            if bbox is not None:
                canvas.configure(scrollregion=bbox)

        def match_content_width(event):
            # Keep the controls fitted to the visible pane horizontally while
            # allowing the content to grow vertically.
            canvas.itemconfigure(window_id, width=max(1, event.width))

        content.bind("<Configure>", update_scroll_region, add="+")
        canvas.bind("<Configure>", match_content_width, add="+")

        # Mark the page/content/canvas so the application-wide mouse-wheel
        # router can find the correct scrollable tab from any child widget.
        page._vertical_scroll_canvas = canvas
        content._vertical_scroll_canvas = canvas
        canvas._vertical_scroll_canvas = canvas

        return page, content

    def _route_tab_mousewheel(self, event):
        """Route mouse-wheel motion to the scrollable control tab under it."""
        widget = getattr(event, "widget", None)
        canvas = None

        while widget is not None:
            canvas = getattr(widget, "_vertical_scroll_canvas", None)
            if canvas is not None:
                break
            widget = getattr(widget, "master", None)

        if canvas is None:
            return None

        bbox = canvas.bbox("all")
        if bbox is None or (bbox[3] - bbox[1]) <= canvas.winfo_height():
            return None

        if getattr(event, "num", None) == 4:      # Linux wheel up
            units = -3
        elif getattr(event, "num", None) == 5:    # Linux wheel down
            units = 3
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return None
            # Windows commonly reports multiples of 120. High-resolution
            # wheels/trackpads may report smaller values, so preserve direction.
            magnitude = max(1, abs(int(delta)) // 120)
            units = -magnitude if delta > 0 else magnitude

        canvas.yview_scroll(units, "units")
        return "break"

    def _build_controls(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Every control tab is scrollable. The public self.tab_* attributes
        # refer to the inner content frames so the existing builders continue
        # to work unchanged.
        self.page_stack, self.tab_stack = self._create_scrollable_notebook_tab("Stacking")
        self.page_corr, self.tab_corr = self._create_scrollable_notebook_tab("Corrections")
        self.page_filter, self.tab_filter = self._create_scrollable_notebook_tab("Spatial Filters")
        self.page_contrast, self.tab_contrast = self._create_scrollable_notebook_tab("Contrast")
        self.page_symmetry, self.tab_symmetry = self._create_scrollable_notebook_tab("Symmetry")
        self.page_fiber, self.tab_fiber = self._create_scrollable_notebook_tab("Fiber Analysis")
        self.page_layer_lines, self.tab_layer_lines = self._create_scrollable_notebook_tab("Layer Lines")
        self.page_roi, self.tab_roi = self._create_scrollable_notebook_tab("ROI / Shapes")
        self.page_comparison, self.tab_comparison = self._create_scrollable_notebook_tab("Compare")
        self.page_performance, self.tab_performance = self._create_scrollable_notebook_tab("Performance")
        self.page_export, self.tab_export = self._create_scrollable_notebook_tab("Export")

        self._build_stacking_tab()
        self._build_corrections_tab()
        self._build_filter_tab()
        self._build_contrast_tab()
        self._build_symmetry_tab()
        self._build_fiber_tab()
        self._build_layer_lines_tab()
        self._build_roi_tab()
        self._build_comparison_tab()
        self._build_performance_tab()
        self._build_export_tab()

        # Use an application-wide wheel binding, but scroll only when the event
        # originated from a widget inside one of the marked notebook pages.
        # This lets the wheel work over entries, buttons, labels, and frames.
        self.bind_all("<MouseWheel>", self._route_tab_mousewheel, add="+")
        self.bind_all("<Button-4>", self._route_tab_mousewheel, add="+")
        self.bind_all("<Button-5>", self._route_tab_mousewheel, add="+")

    def _build_viewer(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=6, pady=(6, 0))

        ttk.Label(toolbar, text="View:").pack(side=tk.LEFT)

        view_combo = ttk.Combobox(
            toolbar,
            textvariable=self.view_var,
            values=[
                "Raw",
                "Corrected",
                "Enhanced",
                "Display",
                "Mask",
                "Background model",
                "Stack",
                "Symmetrized",
                "Asymmetry",
                "Symmetry contributors",
                "Comparison",
            ],
            width=18,
            state="readonly",
        )
        view_combo.pack(side=tk.LEFT, padx=(4, 12))
        view_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.refresh_plot(),
        )

        ttk.Label(toolbar, text="Colormap:").pack(side=tk.LEFT)

        cmap_combo = ttk.Combobox(
            toolbar,
            textvariable=self.cmap_var,
            values=["gray", "inferno", "viridis", "magma", "plasma"],
            width=12,
            state="readonly",
        )
        cmap_combo.pack(side=tk.LEFT, padx=(4, 12))
        cmap_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.refresh_plot(),
        )

        ttk.Button(
            toolbar,
            text="Refresh",
            command=self.refresh_plot,
        ).pack(side=tk.LEFT)

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=parent,
        )
        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
            padx=6,
            pady=6,
        )
        self.canvas.mpl_connect(
            "button_press_event",
            self._on_image_click,
        )
        self.canvas.mpl_connect(
            "button_release_event",
            self._on_image_release,
        )
        self.canvas.mpl_connect(
            "motion_notify_event",
            self._on_image_motion,
        )

    def _section(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(fill=tk.X, padx=8, pady=6)
        return frame

    def _entry_row(self, parent, label, variable, width=10):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(row, text=label).pack(side=tk.LEFT)
        ttk.Entry(
            row,
            textvariable=variable,
            width=width,
        ).pack(side=tk.RIGHT)

        return row

    def _check_entry_row(
        self,
        parent,
        text,
        boolvar,
        valuevar,
        value_label="",
    ):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=6, pady=3)

        ttk.Checkbutton(
            row,
            text=text,
            variable=boolvar,
        ).pack(side=tk.LEFT)

        if value_label:
            ttk.Label(
                row,
                text=value_label,
            ).pack(side=tk.RIGHT)

        ttk.Entry(
            row,
            textvariable=valuevar,
            width=9,
        ).pack(side=tk.RIGHT, padx=3)

    # ============================================================
    # STACKING TAB
    # ============================================================

    def _build_stacking_tab(self):
        f = self._section(
            self.tab_stack,
            "Included frames → combined accumulation",
        )

        row = ttk.Frame(f)
        row.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(row, text="Combination method").pack(side=tk.LEFT)

        ttk.Combobox(
            row,
            textvariable=self.stack_method,
            values=[
                "mean",
                "sum",
                "median",
                "sigma-clipped mean",
                "trimmed mean",
                "winsorized mean",
                "min/max rejected mean",
                "inverse-variance weighted mean",
                "exposure-weighted mean",
                "huber mean",
            ],
            state="readonly",
            width=24,
        ).pack(side=tk.RIGHT)

        exposure = self._section(
            self.tab_stack,
            "Input correction state / exposure normalization",
        )
        r = ttk.Frame(exposure); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r, text="Input correction state").pack(side=tk.LEFT)
        ttk.Combobox(
            r,
            textvariable=self.input_correction_state,
            values=[
                "already dark + flat corrected",
                "raw / uncorrected",
                "custom / mixed",
            ],
            state="readonly",
            width=27,
        ).pack(side=tk.RIGHT)

        r = ttk.Frame(exposure); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r, text="Frame normalization").pack(side=tk.LEFT)
        ttk.Combobox(
            r,
            textvariable=self.stack_normalization_mode,
            values=["none", "exposure"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)
        self._entry_row(exposure, "Reference exposure (s)", self.stack_reference_exposure)

        prow = ttk.Frame(exposure); prow.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(prow, text="Automatic exposure source").pack(side=tk.LEFT)
        policy_box = ttk.Combobox(
            prow,
            textvariable=self.exposure_auto_policy,
            values=["filename only", "filename → header", "header → filename"],
            state="readonly",
            width=20,
        )
        policy_box.pack(side=tk.RIGHT)
        policy_box.bind("<<ComboboxSelected>>", self._exposure_policy_changed)

        vrow = ttk.Frame(exposure); vrow.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(vrow, text="Manual exposure").pack(side=tk.LEFT)
        ttk.Entry(vrow, textvariable=self.exposure_batch_value, width=10).pack(side=tk.RIGHT, padx=(4,0))
        ttk.Combobox(
            vrow, textvariable=self.exposure_batch_unit, values=["s", "ms"],
            state="readonly", width=5,
        ).pack(side=tk.RIGHT)

        erow = ttk.Frame(exposure); erow.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(erow, text="Set selected", command=self.set_selected_exposure).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(erow, text="Set all", command=self.set_all_exposure).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(erow, text="Import CSV…", command=self.import_exposure_csv).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        erow2 = ttk.Frame(exposure); erow2.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(erow2, text="Clear manual", command=self.clear_selected_manual_exposure).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(erow2, text="Re-detect selected", command=self.redetect_selected_exposure).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(erow2, text="Use header selected", command=self.use_header_for_selected_exposure).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        ttk.Label(
            exposure,
            text=(
                "Default: use only explicit filename tokens such as _5s_, _10s_ or _500ms_. "
                "Rayonix/MarCCD header timing is shown as a diagnostic because it can disagree "
                "with the acquisition filename. Manual/CSV values always override automatic values."
            ),
            wraplength=365,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        reg = self._section(
            self.tab_stack,
            "Optional translational registration",
        )

        ttk.Checkbutton(
            reg,
            text="Align every frame to the first included frame",
            variable=self.stack_align,
        ).pack(anchor="w", padx=6, pady=3)

        self._entry_row(
            reg,
            "Maximum allowed shift (pixels)",
            self.stack_max_shift,
        )

        clip = self._section(
            self.tab_stack,
            "Sigma-clipped mean settings",
        )

        self._entry_row(
            clip,
            "Clip threshold (σ)",
            self.stack_sigma,
        )
        self._entry_row(
            clip,
            "Iterations",
            self.stack_iterations,
        )

        robust = self._section(
            self.tab_stack,
            "Additional robust stacking",
        )
        self._entry_row(robust, "Trimmed fraction / side", self.stack_trim_fraction)
        self._entry_row(robust, "Winsor fraction / side", self.stack_winsor_fraction)
        self._entry_row(robust, "Huber delta", self.stack_huber_delta)
        self._entry_row(robust, "Huber iterations", self.stack_huber_iterations)
        self._entry_row(robust, "Noise-weight floor", self.stack_noise_weight_floor)
        ttk.Label(
            robust,
            text=(
                "Inverse-variance weighting is best reserved for comparable exposures "
                "whose noise differs. Trimmed/min-max/Huber methods are useful for "
                "transient spikes or outlying accumulations. Mean remains the default "
                "when frames are clean and comparable."
            ),
            wraplength=360,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        memory = self._section(
            self.tab_stack,
            "Large-image handling",
        )

        self._entry_row(
            memory,
            "Chunk height for median/clip",
            self.stack_chunk_rows,
        )

        ttk.Label(
            memory,
            text=(
                "Mean, sum, and inverse-variance weighting are streamed. Robust "
                "per-pixel methods use a temporary disk-backed array and combine "
                "the CCD image in row chunks to control RAM use."
            ),
            wraplength=360,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        actions = self._section(
            self.tab_stack,
            "Build stack",
        )

        ttk.Button(
            actions,
            text="Build stack from included files",
            command=self.start_stack_worker,
        ).pack(fill=tk.X, padx=6, pady=4)

        self.stack_progress = ttk.Progressbar(
            actions,
            mode="determinate",
            maximum=100,
        )
        self.stack_progress.pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(
            actions,
            textvariable=self.stack_summary_var,
            wraplength=360,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            actions,
            text="Preview stack",
            command=self.preview_stack,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Button(
            actions,
            text="Use stack as current working image",
            command=self.use_stack_as_current,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(
            self.tab_stack,
            text=(
                "The stack is kept separate from the individual TIFFs. "
                "For mean/sum stacks of independent noise, the ideal SNR gain "
                "is approximately √N. Median and sigma-clipped mean are useful "
                "when isolated transient spikes or bad accumulations are present."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=8)

    # ============================================================
    # CORRECTION / FILTER / ANALYSIS / EXPORT TABS
    # ============================================================

    def _build_corrections_tab(self):
        refs = self._section(
            self.tab_corr,
            "Reference images",
        )

        ttk.Button(
            refs,
            text="Load dark TIFF…",
            command=self.load_dark,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Button(
            refs,
            text="Load flat TIFF…",
            command=self.load_flat,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Button(
            refs,
            text="Load blank/background TIFF…",
            command=self.load_background,
        ).pack(fill=tk.X, padx=6, pady=3)

        self.ref_label = ttk.Label(
            refs,
            text="Dark: none\nFlat: none\nBackground: none",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.ref_label.pack(fill=tk.X, padx=6, pady=5)

        f = self._section(
            self.tab_corr,
            "Quantitative corrections",
        )

        ttk.Checkbutton(
            f,
            text="Dark subtraction",
            variable=self.dark_enabled,
        ).pack(anchor="w", padx=6, pady=3)

        ttk.Checkbutton(
            f,
            text="Flat-field correction",
            variable=self.flat_enabled,
        ).pack(anchor="w", padx=6, pady=3)

        self._check_entry_row(
            f,
            "Blank/background subtraction",
            self.background_enabled,
            self.background_scale,
            "scale",
        )

        self._check_entry_row(
            f,
            "Monitor / exposure normalization",
            self.normalize_enabled,
            self.monitor_value,
            "value",
        )

        masks = self._section(
            self.tab_corr,
            "Detector masks",
        )

        self._check_entry_row(
            masks,
            "Hot-pixel mask",
            self.hot_pixels_enabled,
            self.hot_pixel_sigma,
            "σ",
        )

        self._entry_row(
            masks,
            "Hot-pixel neighborhood",
            self.hot_pixel_size,
        )

        self._check_entry_row(
            masks,
            "Saturation mask",
            self.saturation_enabled,
            self.saturation_value,
            "≥",
        )

        beam = self._section(
            self.tab_corr,
            "Beamstop / direct-beam mask",
        )

        ttk.Checkbutton(
            beam,
            text="Enable circular beamstop mask",
            variable=self.beamstop_enabled,
        ).pack(anchor="w", padx=6, pady=3)

        self._entry_row(
            beam,
            "Center X (pixels)",
            self.beamstop_center_x,
        )
        self._entry_row(
            beam,
            "Center Y (pixels)",
            self.beamstop_center_y,
        )
        self._entry_row(
            beam,
            "Radius (pixels)",
            self.beamstop_radius,
        )

        ttk.Button(
            beam,
            text="Click image to set center",
            command=self.start_beam_center_click,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            self.tab_corr,
            text="Apply / Refresh",
            command=self.reprocess,
        ).pack(fill=tk.X, padx=8, pady=8)

    def _build_filter_tab(self):
        noise = self._section(
            self.tab_filter,
            "Noise filtering",
        )

        self._check_entry_row(
            noise,
            "Median filter",
            self.median_filter_enabled,
            self.median_filter_size,
            "px",
        )

        self._check_entry_row(
            noise,
            "Gaussian smoothing",
            self.gaussian_filter_enabled,
            self.gaussian_filter_sigma,
            "σ",
        )

        bg = self._section(
            self.tab_filter,
            "Mathematical background",
        )

        self._check_entry_row(
            bg,
            "Gaussian background subtraction",
            self.gaussian_background_enabled,
            self.gaussian_background_sigma,
            "σ",
        )

        self._check_entry_row(
            bg,
            "Median background subtraction",
            self.median_background_enabled,
            self.median_background_size,
            "px",
        )

        feature = self._section(
            self.tab_filter,
            "Feature enhancement",
        )

        self._check_entry_row(
            feature,
            "High-pass filter",
            self.high_pass_enabled,
            self.high_pass_sigma,
            "σ",
        )

        ttk.Checkbutton(
            feature,
            text="Unsharp mask",
            variable=self.unsharp_enabled,
        ).pack(anchor="w", padx=6, pady=3)

        self._entry_row(
            feature,
            "Unsharp σ",
            self.unsharp_sigma,
        )

        self._entry_row(
            feature,
            "Unsharp amount",
            self.unsharp_amount,
        )

        ttk.Label(
            self.tab_filter,
            text="Tone curves and contrast controls are now in the dedicated Contrast tab.",
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=6)

        ttk.Label(
            self.tab_filter,
            text=(
                "High-pass, unsharp, mathematical-background subtraction "
                "and nonlinear display transforms are kept separate from the "
                "quantitative corrected image."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=6)

        ttk.Button(
            self.tab_filter,
            text="Apply / Refresh",
            command=self.reprocess,
        ).pack(fill=tk.X, padx=8, pady=8)


    def _build_contrast_tab(self):
        tone = self._section(self.tab_contrast, "Tone curve")
        row = ttk.Frame(tone)
        row.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row, text="Display transform").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.display_mode,
            values=["linear", "log", "sqrt", "gamma", "asinh", "hist-eq", "custom curve"],
            state="readonly",
            width=18,
        ).pack(side=tk.RIGHT)
        self._entry_row(tone, "Log gain", self.log_gain)
        self._entry_row(tone, "Gamma", self.gamma)
        self._entry_row(tone, "Asinh strength", self.asinh_strength)

        custom = self._section(self.tab_contrast, "Custom tone curve — display only")
        self.tone_curve_fig = Figure(figsize=(3.45, 2.65), dpi=90)
        self.tone_curve_ax = self.tone_curve_fig.add_subplot(111)
        self.tone_curve_canvas = FigureCanvasTkAgg(self.tone_curve_fig, master=custom)
        self.tone_curve_canvas.get_tk_widget().pack(fill=tk.X, padx=6, pady=4)
        self.tone_curve_canvas.mpl_connect("button_press_event", self._tone_curve_press)
        self.tone_curve_canvas.mpl_connect("motion_notify_event", self._tone_curve_motion)
        self.tone_curve_canvas.mpl_connect("button_release_event", self._tone_curve_release)

        ttk.Label(
            custom,
            text=(
                "Left-click to add/select a point, drag to reshape the curve, and "
                "right-click an interior point to delete it. Input intensity is on X; "
                "display brightness is on Y."
            ),
            wraplength=365,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=(0, 3))

        ttk.Checkbutton(
            custom,
            text="Keep curve monotonic",
            variable=self.tone_curve_monotonic,
        ).pack(anchor="w", padx=6, pady=2)

        curve_buttons1 = ttk.Frame(custom)
        curve_buttons1.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(curve_buttons1, text="Use custom curve", command=self.use_custom_tone_curve).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(curve_buttons1, text="Linear", command=self.tone_curve_linear).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(curve_buttons1, text="S curve", command=self.tone_curve_s_curve).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        curve_buttons2 = ttk.Frame(custom)
        curve_buttons2.pack(fill=tk.X, padx=6, pady=2)
        ttk.Button(curve_buttons2, text="Lift shadows", command=self.tone_curve_lift_shadows).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        ttk.Button(curve_buttons2, text="Compress highlights", command=self.tone_curve_compress_highlights).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(curve_buttons2, text="Save…", command=self.save_tone_curve).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(curve_buttons2, text="Load…", command=self.load_tone_curve).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        levels = self._section(self.tab_contrast, "Black / white levels")
        row2 = ttk.Frame(levels)
        row2.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row2, text="Level strategy").pack(side=tk.LEFT)
        ttk.Combobox(
            row2,
            textvariable=self.contrast_mode,
            values=[
                "legacy percentile",
                "source percentile",
                "manual",
                "robust MAD",
                "full range",
            ],
            state="readonly",
            width=20,
        ).pack(side=tk.RIGHT)
        self._entry_row(levels, "Low percentile", self.percentile_low)
        self._entry_row(levels, "High percentile", self.percentile_high)
        self._entry_row(levels, "Manual black level", self.manual_black)
        self._entry_row(levels, "Manual white level", self.manual_white)
        self._entry_row(levels, "Robust white level (σ)", self.robust_sigma)

        local = self._section(self.tab_contrast, "Adaptive local contrast — display only")
        ttk.Checkbutton(
            local,
            text="Enable local contrast normalization",
            variable=self.local_contrast_enabled,
        ).pack(anchor="w", padx=6, pady=3)
        self._entry_row(local, "Local scale σ (pixels)", self.local_contrast_sigma)
        self._entry_row(local, "Blend strength (0–1)", self.local_contrast_strength)
        self._entry_row(local, "Noise-floor fraction", self.local_contrast_noise_floor)
        ttk.Checkbutton(
            local,
            text="Invert black / white",
            variable=self.invert_display,
        ).pack(anchor="w", padx=6, pady=3)

        actions = self._section(self.tab_contrast, "Inspection / presets")
        ttk.Button(
            actions,
            text="Weak-reflection contrast preset",
            command=self.apply_weak_reflection_preset,
        ).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(
            actions,
            text="Reset contrast defaults",
            command=self.reset_contrast_defaults,
        ).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(
            actions,
            text="Show intensity histogram",
            command=self.show_contrast_histogram,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(
            self.tab_contrast,
            text=(
                "All controls in this tab are display-only. They do not alter the "
                "corrected floating-point diffraction intensity used for profiles, "
                "symmetry averaging, or structural comparison. Custom curves, histogram "
                "equalization and local contrast should be treated as visualization tools."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=8)

        self._draw_tone_curve()

    def _build_symmetry_tab(self):
        info = self._section(
            self.tab_symmetry,
            "Quadrant folding / symmetry averaging",
        )

        ttk.Label(
            info,
            text=(
                "Uses the quantitatively corrected image plus the current beam "
                "center and fiber-axis angle. Output coordinates are aligned with "
                "the equator horizontal and meridian vertical."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=5)

        mode = self._section(
            self.tab_symmetry,
            "Symmetry model",
        )

        row = ttk.Frame(mode)
        row.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row, text="Mode").pack(side=tk.LEFT)

        ttk.Combobox(
            row,
            textvariable=self.symmetry_mode,
            values=[
                "four-quadrant",
                "centrosymmetric",
                "mirror-meridian",
                "mirror-equator",
            ],
            state="readonly",
            width=20,
        ).pack(side=tk.RIGHT)

        row2 = ttk.Frame(mode)
        row2.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(row2, text="Combine by").pack(side=tk.LEFT)

        ttk.Combobox(
            row2,
            textvariable=self.symmetry_statistic,
            values=["mean", "median"],
            state="readonly",
            width=20,
        ).pack(side=tk.RIGHT)

        crop = self._section(
            self.tab_symmetry,
            "Centered symmetric crop",
        )

        self._entry_row(
            crop,
            "Half-width on equator (0 = auto)",
            self.symmetry_half_width,
        )

        self._entry_row(
            crop,
            "Half-height on meridian (0 = auto)",
            self.symmetry_half_height,
        )

        self._entry_row(
            crop,
            "Minimum valid contributors",
            self.symmetry_min_contributors,
        )

        ttk.Label(
            crop,
            text=(
                "Auto mode chooses a centered common region. Masked pixels are "
                "excluded, allowing equivalent unmasked quadrants to contribute "
                "where another quadrant contains a detector defect or gap."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=5)

        action = self._section(
            self.tab_symmetry,
            "Build / inspect",
        )

        ttk.Button(
            action,
            text="Build symmetry average from corrected image",
            command=self.build_symmetry_product,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            action,
            text="Preview symmetrized pattern",
            command=self.preview_symmetry,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Button(
            action,
            text="Preview asymmetry map",
            command=self.preview_asymmetry,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Checkbutton(
            action,
            text="Use symmetrized product for fiber profiles",
            variable=self.use_symmetry_for_profiles,
        ).pack(anchor="w", padx=6, pady=5)

        ttk.Label(
            action,
            textvariable=self.symmetry_summary_var,
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=5)

        ttk.Label(
            self.tab_symmetry,
            text=(
                "Inspect the asymmetry diagnostics before forcing four-quadrant "
                "symmetry. Large disagreement can indicate centering/orientation "
                "errors, tilt, disorder, detector artifacts or genuine asymmetry."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=8)

    def _build_fiber_tab(self):
        auto = self._section(
            self.tab_fiber,
            "Automatic center / axis detection",
        )
        ttk.Button(
            auto,
            text="Detect candidate center + fiber axis",
            command=self.start_axis_detection_worker,
        ).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(
            auto,
            text="Apply detected candidate",
            command=self.apply_axis_detection_result,
        ).pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(
            auto,
            text="Show angle-score diagnostic",
            command=self.show_axis_score_plot,
        ).pack(fill=tk.X, padx=6, pady=3)
        self._entry_row(auto, "Center search radius (px)", self.axis_center_search_radius)
        self._entry_row(auto, "Analysis max dimension", self.axis_analysis_max_dimension)
        self._entry_row(auto, "Coarse angle step (°)", self.axis_coarse_angle_step)
        self._entry_row(auto, "Fine angle step (°)", self.axis_fine_angle_step)
        self._entry_row(auto, "Central exclusion radius", self.axis_central_exclusion)
        ttk.Label(
            auto,
            textvariable=self.axis_detection_summary_var,
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=5)
        ttk.Label(
            auto,
            text=(
                "Detection exploits 180° centrosymmetry for the beam center and "
                "mirror symmetry for the two perpendicular fiber-pattern axes. "
                "The meridian/equator pair is intrinsically ambiguous by 90°, so "
                "the candidate closest to the current fiber-angle estimate is reported."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        center = self._section(
            self.tab_fiber,
            "Beam center",
        )

        self._entry_row(
            center,
            "Center X (pixels)",
            self.center_x,
        )
        self._entry_row(
            center,
            "Center Y (pixels)",
            self.center_y,
        )

        ttk.Button(
            center,
            text="Use image center",
            command=self.use_image_center,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Button(
            center,
            text="Click image to set beam center",
            command=self.start_analysis_center_click,
        ).pack(fill=tk.X, padx=6, pady=3)

        axis = self._section(
            self.tab_fiber,
            "Fiber axis",
        )

        self._entry_row(
            axis,
            "Fiber angle (° from +x)",
            self.fiber_angle,
        )

        ttk.Button(
            axis,
            text="Click two points along fiber axis",
            command=self.start_axis_clicks,
        ).pack(fill=tk.X, padx=6, pady=3)

        prof = self._section(
            self.tab_fiber,
            "Meridian / equator profiles",
        )

        self._entry_row(
            prof,
            "Strip width (pixels)",
            self.strip_width,
        )

        ttk.Button(
            prof,
            text="Plot meridian + equator",
            command=self.plot_profiles,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Button(
            prof,
            text="Export profiles CSV…",
            command=self.export_profiles,
        ).pack(fill=tk.X, padx=6, pady=3)

        peaks = self._section(
            self.tab_fiber,
            "Profile peak detection",
        )

        self._entry_row(
            peaks,
            "Prominence (0 = automatic)",
            self.peak_prominence,
        )

        self._entry_row(
            peaks,
            "Minimum separation (pixels)",
            self.peak_distance,
        )

        ttk.Button(
            peaks,
            text="Find peaks on profiles",
            command=self.plot_profiles_with_peaks,
        ).pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(
            self.tab_fiber,
            text=(
                "Fiber profiles are currently in detector-pixel coordinates. "
                "Detector calibration and q∥/q⊥ conversion are planned for the "
                "next analysis stage."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=8)

    def _build_layer_lines_tab(self):
        setup = self._section(self.tab_layer_lines, "Layer-line ladder")
        self._entry_row(setup, "Lines: equator → anchor", self.layer_line_count)
        self._entry_row(setup, "Anchor offset along fiber (px)", self.layer_line_anchor_offset)
        self._entry_row(setup, "Refine radius (px)", self.layer_line_refine_radius)
        self._entry_row(setup, "Exclude meridian ± (px)", self.layer_line_inner_perp)
        self._entry_row(setup, "Outer integration |perp| (0=auto)", self.layer_line_outer_perp)
        r = ttk.Frame(setup); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="Click anchor", command=self.start_layer_anchor_click).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="Suggest anchor", command=self.suggest_layer_anchor).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(r, text="Fit lines", command=self.fit_layer_lines).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        ttk.Label(
            setup,
            text=(
                "The count includes the equator. Example: 7 means line 0 is the equator and line 6 is the anchor. "
                "The model is mirrored across the equator and each nominal line can be refined locally from the experimental intensity profile."
            ),
            wraplength=365, justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        table = self._section(self.tab_layer_lines, "Fitted layer lines")
        self.layer_tree = ttk.Treeview(table, columns=("n","nominal","measured","resid"), show="headings", height=8)
        for col, txt, width in (("n","n",45),("nominal","Nominal px",85),("measured","Measured px",90),("resid","Residual px",85)):
            self.layer_tree.heading(col, text=txt); self.layer_tree.column(col, width=width, anchor="e")
        self.layer_tree.pack(fill=tk.X, padx=6, pady=3)
        self.layer_tree.bind("<<TreeviewSelect>>", self._layer_tree_selected)
        ttk.Label(table, textvariable=self.layer_line_summary_var, wraplength=365, justify=tk.LEFT).pack(fill=tk.X, padx=6, pady=4)

        prof = self._section(self.tab_layer_lines, "Selected line profile / reflection peaks")
        self._entry_row(prof, "Strip half-width (px)", self.layer_line_half_width)
        self._entry_row(prof, "Max Gaussian peaks", self.layer_line_max_peaks)
        self._entry_row(prof, "Peak prominence (0=auto)", self.layer_line_peak_prominence)
        r = ttk.Frame(prof); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="Show profile", command=self.show_selected_layer_profile).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="Fit peaks", command=self.fit_selected_layer_peaks).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(r, text="Export CSV…", command=self.export_layer_lines_csv).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

    def _build_roi_tab(self):
        draw = self._section(self.tab_roi, "Non-destructive shapes / ROIs")
        ttk.Label(
            draw,
            text="Drag on the detector image to create a shape. Shapes do not alter detector intensities.",
            wraplength=365, justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)
        r = ttk.Frame(draw); r.pack(fill=tk.X, padx=6, pady=3)
        for label, mode in (("Circle","circle"),("Ellipse","ellipse"),("Rectangle","rectangle"),("Line","line")):
            ttk.Button(r, text=label, command=lambda m=mode: self.start_roi_draw(m)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        r = ttk.Frame(draw); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="Delete selected", command=self.delete_selected_roi).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="Clear all", command=self.clear_rois).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(r, text="Save/Load…", command=self.roi_save_load_menu).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        manager = self._section(self.tab_roi, "ROI manager / measurement")
        self.roi_listbox = tk.Listbox(manager, height=9, exportselection=False)
        self.roi_listbox.pack(fill=tk.X, padx=6, pady=3)
        self.roi_listbox.bind("<<ListboxSelect>>", lambda _e: self.measure_selected_roi())
        r = ttk.Frame(manager); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r, text="Measure source").pack(side=tk.LEFT)
        ttk.Combobox(r, textvariable=self.roi_source, values=["Raw","Corrected","Enhanced","Stack"], state="readonly", width=14).pack(side=tk.RIGHT)
        ttk.Button(manager, text="Measure selected ROI", command=self.measure_selected_roi).pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(manager, textvariable=self.roi_summary_var, wraplength=365, justify=tk.LEFT).pack(fill=tk.X, padx=6, pady=4)

    def _build_comparison_tab(self):
        src = self._section(self.tab_comparison, "Comparison images")
        r = ttk.Frame(src); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="A = current corrected", command=lambda: self.set_comparison_from_current("A")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="Load A…", command=lambda: self.load_comparison_file("A")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        r = ttk.Frame(src); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="B = current corrected", command=lambda: self.set_comparison_from_current("B")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="B = stack", command=lambda: self.set_comparison_from_stack("B")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(r, text="Load B…", command=lambda: self.load_comparison_file("B")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))

        cfg = self._section(self.tab_comparison, "Normalization / comparison")
        r = ttk.Frame(cfg); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r, text="Mode").pack(side=tk.LEFT)
        ttk.Combobox(r, textvariable=self.comparison_mode, values=["difference","absolute difference","ratio","overlay"], state="readonly", width=20).pack(side=tk.RIGHT)
        r = ttk.Frame(cfg); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(r, text="Scale B → A").pack(side=tk.LEFT)
        ttk.Combobox(r, textvariable=self.comparison_scale_mode, values=["none","least-squares fit","total intensity","exposure","manual"], state="readonly", width=20).pack(side=tk.RIGHT)
        self._entry_row(cfg, "Manual scale", self.comparison_manual_scale)
        self._entry_row(cfg, "Overlay alpha (B)", self.comparison_alpha)
        self._entry_row(cfg, "Exposure A (s)", self.comparison_exposure_a)
        self._entry_row(cfg, "Exposure B (s)", self.comparison_exposure_b)
        r = ttk.Frame(cfg); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="Build comparison", command=self.build_comparison).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="Side-by-side", command=self.show_comparison_side_by_side).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(r, text="Export report…", command=self.export_comparison_report).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        ttk.Button(cfg, text="Compare selected ROI in A vs B", command=self.compare_selected_roi).pack(fill=tk.X, padx=6, pady=3)
        ttk.Label(cfg, textvariable=self.comparison_summary_var, wraplength=365, justify=tk.LEFT).pack(fill=tk.X, padx=6, pady=4)

    # ============================================================
    # LAYER-LINE ANALYSIS
    # ============================================================

    def start_layer_anchor_click(self):
        if self.results is None:
            messagebox.showinfo("Layer lines", "Load and process an image first.")
            return
        self.click_mode = "layer_anchor"
        self.status_var.set("Click the anchor layer line / meridional reflection in the detector image.")

    def suggest_layer_anchor(self):
        if self.results is None:
            messagebox.showinfo("Layer lines", "Load and process an image first.")
            return
        try:
            value = suggest_anchor_from_axial_profile(
                self.results["corrected"],
                self.center_x.get(), self.center_y.get(), self.fiber_angle.get(),
                min_offset_px=max(10.0, self.layer_line_inner_perp.get() * 1.5),
            )
            self.layer_line_anchor_offset.set(value)
            self.status_var.set(f"Suggested positive anchor offset: {value:.2f} px. Review before fitting.")
        except Exception as exc:
            messagebox.showerror("Layer-line anchor", str(exc))

    def fit_layer_lines(self):
        if self.results is None:
            messagebox.showinfo("Layer lines", "Load and process an image first.")
            return
        try:
            outer = float(self.layer_line_outer_perp.get())
            self.layer_line_result = fit_evenly_spaced_layer_lines(
                self.results["corrected"],
                center_x=float(self.center_x.get()),
                center_y=float(self.center_y.get()),
                fiber_angle_deg=float(self.fiber_angle.get()),
                count_from_equator_to_anchor=int(self.layer_line_count.get()),
                anchor_offset_px=float(self.layer_line_anchor_offset.get()),
                refine_radius_px=float(self.layer_line_refine_radius.get()),
                inner_perp_px=float(self.layer_line_inner_perp.get()),
                outer_perp_px=(outer if outer > 0 else None),
                symmetric=True,
            )
            self._populate_layer_tree()
            r = self.layer_line_result
            abs_res = [abs(line.residual_px) for line in r.lines if line.residual_px is not None and line.index != 0]
            median_res = float(np.median(abs_res)) if abs_res else 0.0
            self.layer_line_summary_var.set(
                f"Spacing {r.nominal_spacing_px:.3f} px; anchor {r.anchor_offset_px:.2f} px; "
                f"median |refinement residual| {median_res:.2f} px."
            )
            self.refresh_plot()
        except Exception as exc:
            messagebox.showerror("Layer-line fitting", str(exc))

    def _populate_layer_tree(self):
        if not hasattr(self, "layer_tree"):
            return
        self.layer_tree.delete(*self.layer_tree.get_children())
        if self.layer_line_result is None:
            return
        for line in self.layer_line_result.lines:
            self.layer_tree.insert(
                "", "end", iid=str(line.index),
                values=(
                    line.index,
                    f"{line.offset_px:.3f}",
                    f"{line.measured_offset_px:.3f}" if line.measured_offset_px is not None else "—",
                    f"{line.residual_px:.3f}" if line.residual_px is not None else "—",
                ),
            )
        if "0" in self.layer_tree.get_children():
            self.layer_tree.selection_set("0")
            self.layer_line_selected_index.set(0)

    def _layer_tree_selected(self, _event=None):
        if not hasattr(self, "layer_tree"):
            return
        selected = self.layer_tree.selection()
        if not selected:
            return
        try:
            self.layer_line_selected_index.set(int(selected[0]))
        except Exception:
            pass

    def _selected_layer_line(self):
        if self.layer_line_result is None:
            raise ValueError("Fit a layer-line model first.")
        idx = int(self.layer_line_selected_index.get())
        for line in self.layer_line_result.lines:
            if line.index == idx:
                return line
        raise ValueError(f"Layer line {idx} is not present in the fitted model.")

    def _selected_layer_profile_data(self):
        line = self._selected_layer_line()
        image = self.results["corrected"]
        offset = line.measured_offset_px if line.measured_offset_px is not None else line.offset_px
        return line, sample_fiber_strip(
            image,
            self.center_x.get(), self.center_y.get(), self.fiber_angle.get(),
            parallel_offset_px=offset,
            half_width_px=float(self.layer_line_half_width.get()),
        )

    def show_selected_layer_profile(self):
        try:
            line, (x, y) = self._selected_layer_profile_data()
        except Exception as exc:
            messagebox.showerror("Layer-line profile", str(exc)); return
        win = tk.Toplevel(self); win.title(f"Layer line {line.index} profile")
        fig = Figure(figsize=(8.5, 5.2), dpi=100); ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=1.0)
        ax.axvline(0, linewidth=0.7, alpha=0.6)
        ax.set_xlabel("Perpendicular coordinate from meridian (pixels)")
        ax.set_ylabel("Mean corrected intensity")
        ax.set_title(f"Layer line {line.index} — offset {line.measured_offset_px:.2f} px")
        fig.tight_layout(); canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas.draw_idle()

    def fit_selected_layer_peaks(self):
        try:
            line, (x, y) = self._selected_layer_profile_data()
            result = fit_profile_peaks(
                x, y,
                max_peaks=max(1, int(self.layer_line_max_peaks.get())),
                prominence=(float(self.layer_line_peak_prominence.get()) or None),
            )
            self.layer_line_peak_results[int(line.index)] = result
        except Exception as exc:
            messagebox.showerror("Layer-line peak fit", str(exc)); return
        win = tk.Toplevel(self); win.title(f"Layer line {line.index} Gaussian peak fit")
        fig = Figure(figsize=(8.5, 6), dpi=100); ax = fig.add_subplot(111)
        ax.plot(result["x"], result["y"], label="Experimental", linewidth=1.0)
        ax.plot(result["x"], result["fit"], label="Fit", linewidth=1.2)
        for peak in result["peaks"]:
            ax.axvline(peak["center_px"], linestyle="--", linewidth=0.7)
        ax.set_xlabel("Perpendicular coordinate (pixels)"); ax.set_ylabel("Corrected intensity")
        ax.set_title(f"Layer {line.index}: {len(result['peaks'])} peak(s), R²={result['r_squared']:.5f}")
        ax.legend(); fig.tight_layout(); canvas=FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True); canvas.draw_idle()

    def export_layer_lines_csv(self):
        if self.layer_line_result is None:
            messagebox.showinfo("Layer lines", "Fit a layer-line model first."); return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile=f"{self._safe_current_stem()}_layer_lines.csv")
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer_index","nominal_offset_px","measured_offset_px","residual_px","profile_intensity"])
            for line in self.layer_line_result.lines:
                writer.writerow([line.index, line.offset_px, line.measured_offset_px, line.residual_px, line.intensity])
        self.status_var.set(f"Saved layer-line table: {Path(path).name}")

    # ============================================================
    # ROI / SHAPE ANNOTATIONS
    # ============================================================

    def start_roi_draw(self, mode):
        if self.raw is None:
            messagebox.showinfo("ROI", "Load an image first."); return
        self.roi_draw_mode = str(mode)
        self.roi_drag_start = None
        self.status_var.set(f"Drag on the image to draw a {mode} ROI.")

    def _refresh_roi_list(self):
        if not hasattr(self, "roi_listbox"): return
        self.roi_listbox.delete(0, tk.END)
        for roi in self.rois:
            label = roi.label or roi.roi_id
            self.roi_listbox.insert(tk.END, f"{roi.roi_id}  {roi.shape_type}  {label}")

    def _selected_roi(self):
        if not hasattr(self, "roi_listbox"): return None
        sel = self.roi_listbox.curselection()
        if not sel: return None
        idx = int(sel[0])
        return self.rois[idx] if 0 <= idx < len(self.rois) else None

    def delete_selected_roi(self):
        sel = self.roi_listbox.curselection() if hasattr(self, "roi_listbox") else ()
        if not sel: return
        del self.rois[int(sel[0])]
        self._refresh_roi_list(); self.refresh_plot(); self.roi_summary_var.set("ROI deleted.")

    def clear_rois(self):
        self.rois.clear(); self._refresh_roi_list(); self.refresh_plot(); self.roi_summary_var.set("All ROIs cleared.")

    def _roi_measure_image(self):
        source = self.roi_source.get()
        if source == "Stack":
            if self.stack_result is None: raise ValueError("No stack has been built.")
            return self.stack_result.image
        if self.results is None: raise ValueError("No working image loaded.")
        return self.results[{"Raw":"raw","Corrected":"corrected","Enhanced":"enhanced"}.get(source,"corrected")]

    def measure_selected_roi(self):
        roi = self._selected_roi()
        if roi is None:
            self.roi_summary_var.set("Select an ROI first."); return
        try:
            m = measure_roi(self._roi_measure_image(), roi)
            self.roi_summary_var.set(
                f"{roi.roi_id}: N={m['pixel_count']:,}; mean={m['mean']:.6g}; median={m['median']:.6g}; "
                f"sum={m['sum']:.6g}; σ={m['std']:.6g}"
            )
        except Exception as exc:
            self.roi_summary_var.set(str(exc))

    def roi_save_load_menu(self):
        save = messagebox.askyesno("ROI file", "Save current ROIs?\n\nChoose No to load an existing ROI JSON file.")
        if save:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile=f"{self._safe_current_stem()}_rois.json")
            if path:
                save_json(path, {"schema":"xrd_roi_shapes","version":"1.0","rois":[r.to_dict() for r in self.rois]})
        else:
            path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
            if path:
                try:
                    data=json.loads(Path(path).read_text(encoding="utf-8")); self.rois=[ROIShape.from_dict(v) for v in data.get("rois",[])]; self.roi_counter=len(self.rois); self._refresh_roi_list(); self.refresh_plot()
                except Exception as exc: messagebox.showerror("ROI load", str(exc))

    # ============================================================
    # IMAGE COMPARISON
    # ============================================================

    def set_comparison_from_current(self, which):
        if self.results is None:
            messagebox.showinfo("Compare", "Load a working image first."); return
        image=np.array(self.results["corrected"], dtype=np.float32, copy=True); label=f"{self.current_label or 'current'} — corrected"
        if which == "A": self.comparison_a, self.comparison_a_label = image, label
        else: self.comparison_b, self.comparison_b_label = image, label
        self.comparison_summary_var.set(f"A: {self.comparison_a_label or '—'}\nB: {self.comparison_b_label or '—'}")

    def set_comparison_from_stack(self, which="B"):
        if self.stack_result is None:
            messagebox.showinfo("Compare", "Build a stack first."); return
        image=np.array(self.stack_result.image, dtype=np.float32, copy=True); label=f"stack — {self.stack_result.settings.method}"
        if which == "A": self.comparison_a, self.comparison_a_label = image, label
        else: self.comparison_b, self.comparison_b_label = image, label
        self.comparison_summary_var.set(f"A: {self.comparison_a_label or '—'}\nB: {self.comparison_b_label or '—'}")

    def load_comparison_file(self, which):
        path=filedialog.askopenfilename(title=f"Load comparison image {which}", filetypes=[("TIFF","*.tif *.tiff"),("All files","*.*")])
        if not path: return
        try: image,_=load_tiff(path)
        except Exception as exc: messagebox.showerror("Comparison load",str(exc)); return
        if which=="A": self.comparison_a,self.comparison_a_label=image,Path(path).name
        else: self.comparison_b,self.comparison_b_label=image,Path(path).name
        exp = None
        try: exp=inspect_tiff_exposure_seconds(path)
        except Exception: pass
        if exp is not None:
            (self.comparison_exposure_a if which=="A" else self.comparison_exposure_b).set(exp)
        self.comparison_summary_var.set(f"A: {self.comparison_a_label or '—'}\nB: {self.comparison_b_label or '—'}")

    def build_comparison(self):
        if self.comparison_a is None or self.comparison_b is None:
            messagebox.showinfo("Compare", "Set both comparison images A and B first."); return
        try:
            self.comparison_result=compare_images(
                self.comparison_a,self.comparison_b,
                mode=self.comparison_mode.get(), scale_mode=self.comparison_scale_mode.get(),
                manual_scale=float(self.comparison_manual_scale.get()), alpha=float(self.comparison_alpha.get()),
                exposure_a=float(self.comparison_exposure_a.get()), exposure_b=float(self.comparison_exposure_b.get()),
            )
            r=self.comparison_result
            corr="—" if r.correlation is None else f"{r.correlation:.6f}"
            nrms="—" if r.normalized_rms_difference is None else f"{100*r.normalized_rms_difference:.3f}%"
            self.comparison_summary_var.set(f"Scale B→A={r.scale_b_to_a:.6g}; correlation={corr}; RMS={r.rms_difference:.6g}; normalized RMS={nrms}")
            self.view_var.set("Comparison"); self.refresh_plot()
        except Exception as exc: messagebox.showerror("Comparison",str(exc))

    def show_comparison_side_by_side(self):
        if self.comparison_a is None or self.comparison_b is None:
            messagebox.showinfo("Compare", "Set both images first."); return
        if self.comparison_a.shape != self.comparison_b.shape:
            messagebox.showerror("Compare", f"Shapes do not match: {self.comparison_a.shape} vs {self.comparison_b.shape}"); return
        win=tk.Toplevel(self); win.title("XRFD pattern comparison — side by side")
        fig=Figure(figsize=(11,5.5),dpi=100); ax1=fig.add_subplot(121); ax2=fig.add_subplot(122)
        from .processing import display_transform
        settings=self.get_processing_settings()
        ax1.imshow(display_transform(self.comparison_a,settings),cmap=self.cmap_var.get(),origin="upper",interpolation="nearest")
        ax2.imshow(display_transform(self.comparison_b,settings),cmap=self.cmap_var.get(),origin="upper",interpolation="nearest")
        ax1.set_title(self.comparison_a_label or "A"); ax2.set_title(self.comparison_b_label or "B")
        for ax in (ax1,ax2): ax.set_xlabel("Detector X (pixels)"); ax.set_ylabel("Detector Y (pixels)")
        fig.tight_layout(); canvas=FigureCanvasTkAgg(fig,master=win); canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True); canvas.draw_idle()

    def compare_selected_roi(self):
        roi = self._selected_roi()
        if roi is None:
            messagebox.showinfo("Compare ROI", "Select an ROI in the ROI / Shapes tab first.")
            return
        if self.comparison_a is None or self.comparison_b is None:
            messagebox.showinfo("Compare ROI", "Set comparison images A and B first.")
            return
        if self.comparison_a.shape != self.comparison_b.shape:
            messagebox.showerror("Compare ROI", "A and B must have the same detector dimensions for linked ROI comparison.")
            return
        try:
            ma = measure_roi(self.comparison_a, roi)
            mb = measure_roi(self.comparison_b, roi)
            ratio = (mb["sum"] / ma["sum"]) if ma["sum"] not in (None, 0) and mb["sum"] is not None else np.nan
            self.comparison_summary_var.set(
                f"{roi.roi_id}: A sum={ma['sum']:.6g}, B sum={mb['sum']:.6g}, B/A={ratio:.6g}; "
                f"A mean={ma['mean']:.6g}, B mean={mb['mean']:.6g}."
            )
        except Exception as exc:
            messagebox.showerror("Compare ROI", str(exc))

    def export_comparison_report(self):
        if self.comparison_result is None:
            messagebox.showinfo("Compare", "Build a comparison first."); return
        path=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")],initialfile="comparison_report.json")
        if path:
            payload={"image_a":self.comparison_a_label,"image_b":self.comparison_b_label,**self.comparison_result.to_dict()}
            save_json(path,payload); self.status_var.set(f"Saved comparison report: {Path(path).name}")

    def _build_performance_tab(self):
        compute = self._section(self.tab_performance, "Compute backend")

        row = ttk.Frame(compute)
        row.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(row, text="Backend").pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.compute_backend,
            values=["auto", "cpu", "gpu"],
            state="readonly",
            width=12,
        ).pack(side=tk.RIGHT)

        ttk.Label(
            compute,
            textvariable=self.gpu_status_var,
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            compute,
            text="Refresh GPU status",
            command=self.refresh_gpu_status,
        ).pack(fill=tk.X, padx=6, pady=4)

        viewer = self._section(self.tab_performance, "Viewer responsiveness")
        self._entry_row(viewer, "Max displayed image dimension", self.viewer_max_dimension)
        ttk.Label(
            viewer,
            text=(
                "Only the on-screen Matplotlib preview is decimated. Full-resolution "
                "TIFF data and quantitative analysis remain unchanged."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        display = self._section(self.tab_performance, "Fast display scaling")
        ttk.Checkbutton(
            display,
            text="Estimate display percentiles from a sampled grid",
            variable=self.fast_percentiles,
        ).pack(anchor="w", padx=6, pady=3)
        self._entry_row(display, "Percentile sample target", self.percentile_sample_target)

        gaussian = self._section(self.tab_performance, "Broad Gaussian background")
        ttk.Checkbutton(
            gaussian,
            text="Use fast reduced-grid Gaussian background",
            variable=self.fast_gaussian_background,
        ).pack(anchor="w", padx=6, pady=3)
        self._entry_row(gaussian, "Downsample factor", self.gaussian_background_downsample)

        median = self._section(self.tab_performance, "Broad median background")
        ttk.Checkbutton(
            median,
            text="Use fast reduced-grid median background",
            variable=self.fast_median_background,
        ).pack(anchor="w", padx=6, pady=3)
        self._entry_row(median, "Downsample factor", self.median_background_downsample)
        ttk.Label(
            median,
            text=(
                "Large exact 2-D median filters are extremely expensive. The fast mode "
                "estimates the broad background on a reduced grid and interpolates it back. "
                "Disable this option when an exact median background is specifically required."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        stack = self._section(self.tab_performance, "Stack / registration")
        self._entry_row(stack, "Registration FFT crop size", self.registration_crop_size)
        self._entry_row(stack, "SciPy FFT workers (-1 = all)", self.fft_workers)
        self._entry_row(stack, "TIFF decode workers (0 = auto)", self.tiff_workers)

        ttk.Label(
            self.tab_performance,
            text=(
                "Auto uses an NVIDIA CUDA GPU through CuPy when available and otherwise "
                "falls back to optimized CPU processing."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=10, pady=8)

        self.after(50, self.refresh_gpu_status)

    def refresh_gpu_status(self):
        info = get_gpu_info()
        if info.available:
            gib = (info.total_memory_bytes or 0) / (1024 ** 3)
            free = (info.free_memory_bytes or 0) / (1024 ** 3)
            self.gpu_status_var.set(
                f"CUDA GPU detected: {info.name}\nVRAM: {free:.1f} GiB free / {gib:.1f} GiB total"
            )
        else:
            self.gpu_status_var.set(
                "GPU acceleration unavailable; optimized CPU path will be used. "
                + (f"({info.error})" if info.error else "")
            )

    def _build_export_tab(self):
        session = self._section(
            self.tab_export,
            "Analysis session",
        )
        r = ttk.Frame(session); r.pack(fill=tk.X, padx=6, pady=3)
        ttk.Button(r, text="Save session…", command=self.save_analysis_session).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(r, text="Load session…", command=self.load_analysis_session).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2,0))
        ttk.Label(
            session,
            text=(
                "Session JSON stores file inclusion/exposure assignments, major processing/display settings, fiber geometry, "
                "layer-line settings and non-destructive ROIs. It references source TIFF paths; it does not embed raw detector data."
            ),
            wraplength=370, justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=4)

        sharing = self._section(
            self.tab_export,
            "PNG sharing / figures",
        )

        ttk.Button(
            sharing,
            text="Save current viewer as PNG…",
            command=self.save_current_view_png,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Checkbutton(
            sharing,
            text="Include axes / tick labels",
            variable=self.png_include_axes,
        ).pack(anchor="w", padx=6, pady=2)

        ttk.Checkbutton(
            sharing,
            text="Include title",
            variable=self.png_include_title,
        ).pack(anchor="w", padx=6, pady=2)

        self._entry_row(
            sharing,
            "PNG resolution (DPI)",
            self.png_dpi,
        )

        ttk.Label(
            sharing,
            text=(
                "PNG is intended for sharing, figures and presentations. "
                "Keep the floating-point TIFF as the quantitative data product."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=5)

        current = self._section(
            self.tab_export,
            "Current working image",
        )

        ttk.Button(
            current,
            text="Save corrected TIFF…",
            command=self.save_corrected,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            current,
            text="Save enhanced TIFF…",
            command=self.save_enhanced,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            current,
            text="Save display PNG…",
            command=self.save_display_png,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            current,
            text="Save mask TIFF…",
            command=self.save_mask,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            current,
            text="Save processing metadata…",
            command=self.save_metadata,
        ).pack(fill=tk.X, padx=6, pady=4)

        provenance = self._section(
            self.tab_export,
            "Provenance / detector-coordinate record",
        )

        ttk.Checkbutton(
            provenance,
            text="Write JSON + annotated TXT sidecars with corrected TIFF",
            variable=self.provenance_auto_sidecars,
        ).pack(anchor="w", padx=6, pady=3)

        self._entry_row(
            provenance,
            "Geometry calibration label",
            self.provenance_calibration_label,
            width=16,
        )

        self._entry_row(
            provenance,
            "Expected detector rows",
            self.provenance_expected_rows,
        )

        self._entry_row(
            provenance,
            "Expected detector columns",
            self.provenance_expected_columns,
        )

        ttk.Button(
            provenance,
            text="Save provenance bundle (JSON + TXT)…",
            command=self.save_provenance_bundle,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Label(
            provenance,
            text=(
                "The JSON is the canonical machine record. The TXT contains the same "
                "typed values plus explanatory comments and can be parsed back by the "
                "toolkit. The calibration label records intended use; it does not by "
                "itself verify the contents of a PONI file."
            ),
            wraplength=370,
            justify=tk.LEFT,
        ).pack(fill=tk.X, padx=6, pady=5)

        symmetry = self._section(
            self.tab_export,
            "Symmetry products",
        )

        ttk.Button(
            symmetry,
            text="Save symmetrized TIFF…",
            command=self.save_symmetry_tiff,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            symmetry,
            text="Save symmetrized PNG…",
            command=self.save_symmetry_png,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            symmetry,
            text="Save asymmetry TIFF…",
            command=self.save_asymmetry_tiff,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            symmetry,
            text="Save symmetry report…",
            command=self.save_symmetry_report,
        ).pack(fill=tk.X, padx=6, pady=4)

        stack = self._section(
            self.tab_export,
            "Stack products",
        )

        ttk.Button(
            stack,
            text="Save stacked TIFF…",
            command=self.save_stack_tiff,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            stack,
            text="Save stack PNG…",
            command=self.save_stack_png,
        ).pack(fill=tk.X, padx=6, pady=4)

        ttk.Button(
            stack,
            text="Save stack report…",
            command=self.save_stack_report,
        ).pack(fill=tk.X, padx=6, pady=4)

    # ============================================================
    # FILE PANE / INCLUSION STATE
    # ============================================================

    def _apply_exposure_policy_to_entry(self, entry, preserve_manual=True):
        if preserve_manual:
            source = str(entry.get("exposure_source", ""))
            if source.lower().startswith("manual") or source.lower() == "session" or source.upper().startswith("CSV"):
                return
        info = entry.get("exposure_info") or {}
        exposure, source = select_exposure_from_info(info, self.exposure_auto_policy.get())
        entry["exposure"] = exposure
        entry["exposure_source"] = source

    def _exposure_status(self, entry):
        info = entry.get("exposure_info") or {}
        source = str(entry.get("exposure_source") or "")
        if info.get("exposure_mismatch"):
            return "MISMATCH"
        if entry.get("exposure") is None:
            if info.get("header_exposure_seconds") is not None:
                return "HEADER ONLY"
            return "NEEDS VALUE"
        if source.lower().startswith("manual"):
            return "MANUAL"
        if source.upper().startswith("CSV"):
            return "CSV"
        return "OK"

    def _new_entry(self, path, included=True):
        path = Path(path)
        try:
            shape = tuple(inspect_tiff_output_shape(path))
        except Exception:
            shape = None
        try:
            exposure_info = inspect_tiff_exposure_info(path)
        except Exception:
            exposure_info = {
                "filename_exposure_seconds": None,
                "header_exposure_seconds": None,
                "header_exposure_source": None,
                "marccd_integration_seconds": None,
                "exposure_mismatch": False,
            }
        entry = {
            "path": path,
            "included": bool(included),
            "shape": shape,
            "exposure": None,
            "exposure_source": None,
            "exposure_info": exposure_info,
            "mean": None,
            "max": None,
            "corr": None,
        }
        self._apply_exposure_policy_to_entry(entry, preserve_manual=False)
        return entry

    def _rebuild_file_tree(self):
        self.file_tree.delete(*self.file_tree.get_children())

        for idx, entry in enumerate(self.folder_entries):
            use = "✓" if entry["included"] else "–"

            mean = (
                f"{entry['mean']:.4g}"
                if entry["mean"] is not None
                else "—"
            )

            maximum = (
                f"{entry['max']:.4g}"
                if entry["max"] is not None
                else "—"
            )

            corr = (
                f"{entry['corr']:.3f}"
                if entry["corr"] is not None and np.isfinite(entry["corr"])
                else "—"
            )

            shape = entry.get("shape")
            shape_text = (
                f"{shape[1]}×{shape[0]}"
                if shape is not None
                else "?"
            )
            exposure = entry.get("exposure")
            exposure_text = f"{exposure:.6g}" if exposure is not None else "—"
            source_text = entry.get("exposure_source") or "—"
            info = entry.get("exposure_info") or {}
            header_exposure = info.get("header_exposure_seconds")
            header_text = f"{header_exposure:.6g}" if header_exposure is not None else "—"
            qc_text = self._exposure_status(entry)

            tags = []
            if not entry["included"]:
                tags.append("excluded")
            if (
                self.folder_common_shape is not None
                and shape is not None
                and shape != self.folder_common_shape
            ):
                tags.append("shape_mismatch")
            if info.get("exposure_mismatch"):
                tags.append("exposure_mismatch")
            elif entry.get("exposure") is None:
                tags.append("exposure_unknown")

            self.file_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    use,
                    entry["path"].name,
                    shape_text,
                    exposure_text,
                    source_text,
                    header_text,
                    qc_text,
                    mean,
                    maximum,
                    corr,
                ),
                tags=tuple(tags),
            )

        self._update_include_count()

    def _update_tree_row(self, idx):
        if idx < 0 or idx >= len(self.folder_entries):
            return

        entry = self.folder_entries[idx]

        use = "✓" if entry["included"] else "–"

        mean = (
            f"{entry['mean']:.4g}"
            if entry["mean"] is not None
            else "—"
        )

        maximum = (
            f"{entry['max']:.4g}"
            if entry["max"] is not None
            else "—"
        )

        corr = (
            f"{entry['corr']:.3f}"
            if entry["corr"] is not None and np.isfinite(entry["corr"])
            else "—"
        )

        shape = entry.get("shape")
        shape_text = (
            f"{shape[1]}×{shape[0]}"
            if shape is not None
            else "?"
        )
        exposure = entry.get("exposure")
        exposure_text = f"{exposure:.6g}" if exposure is not None else "—"
        source_text = entry.get("exposure_source") or "—"
        info = entry.get("exposure_info") or {}
        header_exposure = info.get("header_exposure_seconds")
        header_text = f"{header_exposure:.6g}" if header_exposure is not None else "—"
        qc_text = self._exposure_status(entry)

        tags = []
        if not entry["included"]:
            tags.append("excluded")
        if (
            self.folder_common_shape is not None
            and shape is not None
            and shape != self.folder_common_shape
        ):
            tags.append("shape_mismatch")
        if info.get("exposure_mismatch"):
            tags.append("exposure_mismatch")
        elif entry.get("exposure") is None:
            tags.append("exposure_unknown")

        self.file_tree.item(
            str(idx),
            values=(
                use,
                entry["path"].name,
                shape_text,
                exposure_text,
                source_text,
                header_text,
                qc_text,
                mean,
                maximum,
                corr,
            ),
            tags=tuple(tags),
        )

        self._update_include_count()

    def _update_include_count(self):
        included = sum(
            1 for entry in self.folder_entries
            if entry["included"]
        )

        self.include_count_label.configure(
            text=f"{included} / {len(self.folder_entries)} included"
        )

    def _row_id_at_event(self, event):
        row_id = self.file_tree.identify_row(event.y)
        return row_id if row_id else None

    def _tree_plain_click(self, event):
        row_id = self._row_id_at_event(event)
        if row_id:
            self._tree_selection_anchor = row_id
        # Return None so the native single-click selection/preview proceeds.
        return None

    def _tree_ctrl_click(self, event):
        row_id = self._row_id_at_event(event)
        if not row_id:
            return "break"
        selected = set(self.file_tree.selection())
        if row_id in selected:
            selected.remove(row_id)
        else:
            selected.add(row_id)
        self.file_tree.selection_set(tuple(sorted(selected, key=int)))
        self.file_tree.focus(row_id)
        self._tree_selection_anchor = row_id
        self._preview_focused_tree_item()
        return "break"

    def _tree_range_ids(self, anchor_id, target_id):
        children = list(self.file_tree.get_children(""))
        if not children or anchor_id not in children or target_id not in children:
            return [target_id] if target_id else []
        a = children.index(anchor_id)
        b = children.index(target_id)
        lo, hi = sorted((a, b))
        return children[lo:hi + 1]

    def _tree_shift_click(self, event):
        row_id = self._row_id_at_event(event)
        if not row_id:
            return "break"
        anchor = self._tree_selection_anchor or self.file_tree.focus() or row_id
        range_ids = self._tree_range_ids(anchor, row_id)
        self.file_tree.selection_set(range_ids)
        self.file_tree.focus(row_id)
        self._preview_focused_tree_item()
        return "break"

    def _tree_ctrl_shift_click(self, event):
        """Add the anchor→clicked range without clearing existing selection."""
        row_id = self._row_id_at_event(event)
        if not row_id:
            return "break"
        anchor = self._tree_selection_anchor or self.file_tree.focus() or row_id
        range_ids = self._tree_range_ids(anchor, row_id)
        selected = list(self.file_tree.selection())
        merged = set(selected)
        merged.update(range_ids)
        self.file_tree.selection_set(tuple(sorted(merged, key=int)))
        self.file_tree.focus(row_id)
        self._preview_focused_tree_item()
        return "break"

    def _selected_tree_indices(self):
        out = []

        for iid in self.file_tree.selection():
            try:
                out.append(int(iid))
            except ValueError:
                pass

        return out

    def include_selected(self):
        for idx in self._selected_tree_indices():
            self.folder_entries[idx]["included"] = True
            self._update_tree_row(idx)

    def exclude_selected(self):
        for idx in self._selected_tree_indices():
            self.folder_entries[idx]["included"] = False
            self._update_tree_row(idx)

    def include_all(self):
        for idx, entry in enumerate(self.folder_entries):
            entry["included"] = True
            self._update_tree_row(idx)

    def exclude_all(self):
        for idx, entry in enumerate(self.folder_entries):
            entry["included"] = False
            self._update_tree_row(idx)

    def invert_inclusion(self):
        for idx, entry in enumerate(self.folder_entries):
            entry["included"] = not entry["included"]
            self._update_tree_row(idx)

    def use_highlighted_only(self):
        selected = set(self._selected_tree_indices())

        for idx, entry in enumerate(self.folder_entries):
            entry["included"] = idx in selected
            self._update_tree_row(idx)

    def _toggle_highlighted_inclusion(self, _event=None):
        selected = self._selected_tree_indices()

        if not selected:
            return "break"

        all_included = all(
            self.folder_entries[idx]["included"]
            for idx in selected
        )

        new_value = not all_included

        for idx in selected:
            self.folder_entries[idx]["included"] = new_value
            self._update_tree_row(idx)

        return "break"

    def _tree_double_click(self, event):
        row_id = self.file_tree.identify_row(event.y)

        if not row_id:
            return

        idx = int(row_id)
        entry = self.folder_entries[idx]
        entry["included"] = not entry["included"]
        self._update_tree_row(idx)

    def _preview_focused_tree_item(self, _event=None):
        focus = self.file_tree.focus()

        if not focus:
            return

        try:
            idx = int(focus)
        except ValueError:
            return

        if 0 <= idx < len(self.folder_entries):
            path = self.folder_entries[idx]["path"]
            self._load_main_image(path, keep_folder=True)

    def _manual_exposure_seconds(self):
        try:
            value = float(self.exposure_batch_value.get())
        except Exception:
            value = 0.0
        unit = str(self.exposure_batch_unit.get() or "s").strip().lower()
        if unit == "ms":
            value /= 1000.0
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Exposure time must be a positive number.")
        return float(value)

    def _exposure_policy_changed(self, _event=None):
        changed = 0
        for idx, entry in enumerate(self.folder_entries):
            source = str(entry.get("exposure_source") or "")
            if source.lower().startswith("manual") or source.upper().startswith("CSV") or source.lower() == "session":
                continue
            self._apply_exposure_policy_to_entry(entry, preserve_manual=False)
            self._update_tree_row(idx)
            changed += 1
        self.status_var.set(
            f"Exposure policy '{self.exposure_auto_policy.get()}' applied to {changed} non-manual frame(s)."
        )

    def set_selected_exposure(self):
        try:
            value = self._manual_exposure_seconds()
        except Exception as exc:
            messagebox.showerror("Exposure", str(exc))
            return
        indices = self._selected_tree_indices()
        if not indices:
            messagebox.showinfo("Exposure", "Select one or more TIFF rows first.")
            return
        for idx in indices:
            self.folder_entries[idx]["exposure"] = value
            self.folder_entries[idx]["exposure_source"] = "manual"
            self._update_tree_row(idx)
        self.status_var.set(f"Set exposure={value:g} s for {len(indices)} selected frame(s).")

    def set_all_exposure(self):
        try:
            value = self._manual_exposure_seconds()
        except Exception as exc:
            messagebox.showerror("Exposure", str(exc))
            return
        for idx, entry in enumerate(self.folder_entries):
            entry["exposure"] = value
            entry["exposure_source"] = "manual"
            self._update_tree_row(idx)
        self.status_var.set(f"Set exposure={value:g} s for all {len(self.folder_entries)} frame(s).")

    def clear_selected_manual_exposure(self):
        indices = self._selected_tree_indices()
        if not indices:
            messagebox.showinfo("Exposure", "Select one or more TIFF rows first.")
            return
        changed = 0
        for idx in indices:
            entry = self.folder_entries[idx]
            source = str(entry.get("exposure_source") or "")
            if source.lower().startswith("manual") or source.upper().startswith("CSV") or source.lower() == "session":
                entry["exposure"] = None
                entry["exposure_source"] = None
                self._apply_exposure_policy_to_entry(entry, preserve_manual=False)
                self._update_tree_row(idx)
                changed += 1
        self.status_var.set(f"Cleared manual/CSV/session exposure on {changed} selected frame(s).")

    def redetect_selected_exposure(self):
        indices = self._selected_tree_indices()
        if not indices:
            messagebox.showinfo("Exposure", "Select one or more TIFF rows first.")
            return
        for idx in indices:
            entry = self.folder_entries[idx]
            try:
                entry["exposure_info"] = inspect_tiff_exposure_info(entry["path"])
            except Exception:
                pass
            # Re-detect deliberately clears any manual value on these rows.
            entry["exposure"] = None
            entry["exposure_source"] = None
            self._apply_exposure_policy_to_entry(entry, preserve_manual=False)
            self._update_tree_row(idx)
        self.status_var.set(f"Re-detected exposure candidates for {len(indices)} selected frame(s).")

    def use_header_for_selected_exposure(self):
        indices = self._selected_tree_indices()
        if not indices:
            messagebox.showinfo("Exposure", "Select one or more TIFF rows first.")
            return
        applied = 0
        missing = 0
        for idx in indices:
            entry = self.folder_entries[idx]
            info = entry.get("exposure_info") or {}
            value = info.get("header_exposure_seconds")
            if value is None:
                missing += 1
                continue
            entry["exposure"] = float(value)
            entry["exposure_source"] = "manual header: " + (info.get("header_exposure_source") or "header")
            self._update_tree_row(idx)
            applied += 1
        self.status_var.set(
            f"Used header exposure for {applied} selected frame(s); {missing} had no header exposure."
        )

    def import_exposure_csv(self):
        path = filedialog.askopenfilename(
            title="Import exposure times",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            mapping = {}
            with open(path, "r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError("CSV has no header row.")
                normalized = {name.lower().strip(): name for name in reader.fieldnames}
                file_key = next((normalized[k] for k in ("file", "filename", "path") if k in normalized), None)
                exp_key = next((normalized[k] for k in ("exposure", "exposure_s", "exposure_seconds", "integration_time") if k in normalized), None)
                if file_key is None or exp_key is None:
                    raise ValueError("CSV must contain filename/file and exposure/exposure_seconds columns.")
                for row in reader:
                    name = Path(str(row[file_key]).strip()).name
                    value = float(row[exp_key])
                    if np.isfinite(value) and value > 0:
                        mapping[name] = value
            matched = 0
            for idx, entry in enumerate(self.folder_entries):
                if entry["path"].name in mapping:
                    entry["exposure"] = mapping[entry["path"].name]
                    entry["exposure_source"] = f"CSV:{Path(path).name}"
                    self._update_tree_row(idx)
                    matched += 1
            self.status_var.set(f"Imported exposure times for {matched} frame(s).")
        except Exception as exc:
            messagebox.showerror("Exposure CSV", str(exc))

    def included_paths(self):
        return [
            entry["path"]
            for entry in self.folder_entries
            if entry["included"]
        ]

    # ============================================================
    # TIFF LOADING
    # ============================================================

    def load_single_tiff(self):
        path = filedialog.askopenfilename(
            title="Open XRD TIFF",
            filetypes=[
                ("TIFF images", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        self.folder_path = Path(path).parent
        self.folder_entries = [self._new_entry(Path(path), included=True)]
        self.folder_common_shape = self.folder_entries[0].get("shape")

        self.folder_label.configure(
            text=f"Single TIFF:\n{Path(path).name}"
        )

        self._rebuild_file_tree()
        self.file_tree.selection_set("0")
        self.file_tree.focus("0")

        self._load_main_image(Path(path), keep_folder=True)

    def load_folder(self):
        folder = filedialog.askdirectory(
            title="Choose folder containing TIFF accumulations"
        )

        if not folder:
            return

        paths = list_tiffs(folder)

        if not paths:
            messagebox.showinfo(
                "No TIFF files",
                "No .tif or .tiff files were found in that folder.",
            )
            return

        self.folder_path = Path(folder)
        self.folder_entries = [
            self._new_entry(path, included=True)
            for path in paths
        ]

        valid_shapes = [
            entry["shape"] for entry in self.folder_entries
            if entry.get("shape") is not None
        ]
        if valid_shapes:
            from collections import Counter
            self.folder_common_shape = Counter(valid_shapes).most_common(1)[0][0]
        else:
            self.folder_common_shape = None

        self.folder_label.configure(
            text=(
                f"{self.folder_path}\n"
                f"{len(paths)} TIFF file(s)"
            )
        )

        self._rebuild_file_tree()

        self.file_tree.selection_set("0")
        self.file_tree.focus("0")
        self._load_main_image(paths[0], keep_folder=True)

        self.stack_result = None
        self.stack_summary_var.set("No stack has been built.")
        self.stack_progress["value"] = 0

        self.status_var.set(
            f"Loaded folder with {len(paths)} TIFF file(s)."
        )

    def _load_main_image(self, path, keep_folder=False):
        try:
            image, meta = load_tiff(path)

            self.current_path = Path(path)
            self.current_label = self.current_path.name
            self.current_source_kind = "tiff"
            self.raw = image
            self.processing_engine.reset()

            h, w = image.shape

            if self.center_x.get() == 0 and self.center_y.get() == 0:
                self.center_x.set((w - 1) / 2)
                self.center_y.set((h - 1) / 2)

            if (
                self.beamstop_center_x.get() == 0
                and self.beamstop_center_y.get() == 0
            ):
                self.beamstop_center_x.set((w - 1) / 2)
                self.beamstop_center_y.set((h - 1) / 2)

            self.reprocess(show_errors=False)

            self.status_var.set(
                f"Previewing {self.current_path.name} "
                f"({w} × {h}, {meta['dtype']})"
            )

        except Exception as exc:
            messagebox.showerror(
                "Load error",
                str(exc),
            )

    def load_dark(self):
        self._load_reference("dark")

    def load_flat(self):
        self._load_reference("flat")

    def load_background(self):
        self._load_reference("background")

    def _load_reference(self, kind):
        path = filedialog.askopenfilename(
            title=f"Load {kind} TIFF",
            filetypes=[
                ("TIFF images", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            image, _meta = load_tiff(path)
            setattr(self, kind, image)
            setattr(self, f"{kind}_path", Path(path))
            self.processing_engine.reset()

            self._update_ref_label()
            self.reprocess(show_errors=False)

        except Exception as exc:
            messagebox.showerror(
                "Reference image error",
                str(exc),
            )

    def _update_ref_label(self):
        def nm(path):
            return path.name if path else "none"

        self.ref_label.configure(
            text=(
                f"Dark: {nm(self.dark_path)}\n"
                f"Flat: {nm(self.flat_path)}\n"
                f"Background: {nm(self.background_path)}"
            )
        )

    # ============================================================
    # STACK WORKER
    # ============================================================

    def get_stack_settings(self):
        exposures = {}
        exposure_sources = {}
        exposure_candidates = {}
        for entry in self.folder_entries:
            keys = [str(entry["path"]), entry["path"].name]
            try:
                keys.append(str(entry["path"].resolve()))
            except Exception:
                pass
            for key in keys:
                exposure_sources[key] = entry.get("exposure_source") or "unknown"
                exposure_candidates[key] = dict(entry.get("exposure_info") or {})
            if entry.get("exposure") is not None:
                value = float(entry["exposure"])
                for key in keys:
                    exposures[key] = value
        return StackSettings(
            method=self.stack_method.get(),
            align=self.stack_align.get(),
            max_shift_pixels=int(self.stack_max_shift.get()),
            sigma_clip_threshold=float(self.stack_sigma.get()),
            sigma_clip_iterations=int(self.stack_iterations.get()),
            trim_fraction=float(self.stack_trim_fraction.get()),
            winsor_fraction=float(self.stack_winsor_fraction.get()),
            huber_delta=float(self.stack_huber_delta.get()),
            huber_iterations=int(self.stack_huber_iterations.get()),
            noise_weight_floor=float(self.stack_noise_weight_floor.get()),
            chunk_rows=int(self.stack_chunk_rows.get()),
            normalization_mode=self.stack_normalization_mode.get(),
            reference_exposure_seconds=float(self.stack_reference_exposure.get()),
            frame_exposures=exposures,
            frame_exposure_sources=exposure_sources,
            frame_exposure_candidates=exposure_candidates,
            input_correction_state=self.input_correction_state.get(),
            compute_backend=self.compute_backend.get(),
            fft_workers=int(self.fft_workers.get()),
            registration_crop_size=int(self.registration_crop_size.get()),
            tiff_workers=int(self.tiff_workers.get()),
        )

    def start_stack_worker(self):
        if self.worker_active:
            messagebox.showinfo(
                "Stacking in progress",
                "A stack is already being built.",
            )
            return

        paths = self.included_paths()

        if not paths:
            messagebox.showinfo(
                "No included files",
                "Include at least one TIFF in the left pane.",
            )
            return

        settings = self.get_stack_settings()

        if settings.normalization_mode == "exposure":
            missing = [
                p.name for p in paths
                if not any(
                    key in settings.frame_exposures
                    for key in (str(p), str(p.resolve()), p.name)
                )
            ]
            if missing:
                shown = "\n".join(f"• {name}" for name in missing[:12])
                if len(missing) > 12:
                    shown += f"\n… and {len(missing)-12} more."
                messagebox.showerror(
                    "Missing exposure times",
                    "Exposure normalization is enabled, but these included frames have no exposure time:\n\n" + shown,
                )
                return
            if float(settings.reference_exposure_seconds) <= 0:
                messagebox.showerror(
                    "Exposure normalization",
                    "Reference exposure must be positive.",
                )
                return

            included_set = {str(p) for p in paths}
            mismatches = [
                entry for entry in self.folder_entries
                if str(entry["path"]) in included_set
                and (entry.get("exposure_info") or {}).get("exposure_mismatch")
            ]
            if mismatches:
                lines = []
                for entry in mismatches[:10]:
                    info = entry.get("exposure_info") or {}
                    lines.append(
                        f"• {entry['path'].name}: selected {entry.get('exposure'):g} s "
                        f"({entry.get('exposure_source') or 'unknown'}), "
                        f"header {info.get('header_exposure_seconds'):g} s"
                    )
                if len(mismatches) > 10:
                    lines.append(f"… and {len(mismatches)-10} more.")
                proceed = messagebox.askyesno(
                    "Exposure mismatch warning",
                    "Filename/manual and detector-header exposure values disagree for "
                    "one or more included frames:\n\n" + "\n".join(lines) +
                    "\n\nContinue using the selected exposure values shown in the file table?",
                )
                if not proceed:
                    return

        # Validate dimensions before robust stack methods allocate a disk-backed
        # array.  If a thumbnail/processed TIFF is mixed into the folder, offer
        # to exclude it rather than failing deep inside the worker thread.
        try:
            preflight = preflight_stack_shapes(paths)
        except Exception as exc:
            messagebox.showerror("Stack selection error", str(exc))
            return

        if preflight.incompatible:
            expected_h, expected_w = preflight.expected_shape
            details = "\n".join(
                f"• {p.name}: {shape[1]} × {shape[0]}"
                for p, shape in preflight.incompatible[:12]
            )
            if len(preflight.incompatible) > 12:
                details += f"\n… and {len(preflight.incompatible)-12} more."

            answer = messagebox.askyesno(
                "Different TIFF dimensions detected",
                (
                    f"The included TIFFs do not all have the same image size.\n\n"
                    f"Main stack size: {expected_w} × {expected_h} pixels.\n\n"
                    f"Incompatible file(s):\n{details}\n\n"
                    "Exclude the incompatible files and continue?"
                ),
            )
            if not answer:
                return

            bad = {str(p.resolve()) for p, _shape in preflight.incompatible}
            for idx, entry in enumerate(self.folder_entries):
                if str(entry["path"].resolve()) in bad:
                    entry["included"] = False
                    self._update_tree_row(idx)

            paths = self.included_paths()
            if not paths:
                messagebox.showinfo(
                    "No compatible files",
                    "No compatible TIFF files remain selected for stacking.",
                )
                return

        self.worker_active = True
        self.stack_progress["value"] = 0
        self.stack_summary_var.set(
            f"Building {settings.method} stack from {len(paths)} frame(s)…"
        )
        self.status_var.set("Building stack…")

        def progress_callback(info):
            self.worker_queue.put(("progress", info))

        def worker():
            try:
                result = build_stack(
                    paths,
                    settings,
                    progress_callback=progress_callback,
                )
                self.worker_queue.put(("done", result))
            except Exception:
                self.worker_queue.put(
                    ("error", traceback.format_exc())
                )

        thread = threading.Thread(
            target=worker,
            daemon=True,
        )
        thread.start()

        self.after(100, self._poll_worker_queue)

    def _poll_worker_queue(self):
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()

                if kind == "progress":
                    total = max(payload.get("total", 1), 1)
                    current = payload.get("current", 0)
                    percent = 100 * current / total

                    self.stack_progress["value"] = percent
                    self.status_var.set(
                        payload.get("message", "Building stack…")
                    )

                elif kind == "done":
                    self.worker_active = False
                    self.stack_result = payload
                    self.stack_progress["value"] = 100

                    self._apply_stack_stats_to_tree()
                    self._update_stack_summary()

                    self.view_var.set("Stack")
                    self.refresh_plot()

                    self.status_var.set("Stack complete.")

                elif kind == "error":
                    self.worker_active = False
                    self.stack_progress["value"] = 0
                    self.stack_summary_var.set("Stack failed.")

                    messagebox.showerror(
                        "Stacking error",
                        payload,
                    )

        except queue.Empty:
            pass

        if self.worker_active:
            self.after(100, self._poll_worker_queue)

    def _apply_stack_stats_to_tree(self):
        if self.stack_result is None:
            return

        by_path = {
            str(Path(stat.path).resolve()): stat
            for stat in self.stack_result.frame_stats
        }

        for idx, entry in enumerate(self.folder_entries):
            key = str(entry["path"].resolve())
            stat = by_path.get(key)

            if stat is not None:
                entry["mean"] = stat.mean
                entry["max"] = stat.maximum
                entry["corr"] = stat.correlation_to_reference
                self._update_tree_row(idx)

    def _update_stack_summary(self):
        if self.stack_result is None:
            self.stack_summary_var.set("No stack has been built.")
            return

        summary = self.stack_result.summary()

        text = (
            f"{summary['frame_count']} frame(s) combined by "
            f"{summary['method']}.\n"
            f"Shape: {summary['shape'][1]} × {summary['shape'][0]} pixels.\n"
            f"Registration: "
            f"{'enabled' if summary['aligned'] else 'off'}."
        )

        gain = summary["approximate_snr_gain_vs_one_frame"]

        if gain is not None:
            text += (
                f"\nIdeal random-noise SNR gain: ~{gain:.2f}×."
            )

        self.stack_summary_var.set(text)

    def preview_stack(self):
        if self.stack_result is None:
            messagebox.showinfo(
                "No stack",
                "Build a stack first.",
            )
            return

        self.view_var.set("Stack")
        self.refresh_plot()

    def use_stack_as_current(self):
        if self.stack_result is None:
            messagebox.showinfo(
                "No stack",
                "Build a stack first.",
            )
            return

        self.raw = np.asarray(
            self.stack_result.image,
            dtype=np.float32,
        ).copy()
        self.processing_engine.reset()

        self.current_path = None
        self.current_source_kind = "stack"
        self.current_label = (
            f"STACK ({len(self.stack_result.frame_stats)} frames, "
            f"{self.stack_result.settings.method})"
        )

        h, w = self.raw.shape
        self.center_x.set((w - 1) / 2)
        self.center_y.set((h - 1) / 2)

        self.reprocess(show_errors=False)
        self.view_var.set("Display")
        self.refresh_plot()

        self.status_var.set(
            "Stack is now the current working image."
        )


    # ============================================================
    # SYMMETRY / QUADRANT FOLDING
    # ============================================================

    def get_symmetry_settings(self):
        return SymmetrySettings(
            mode=self.symmetry_mode.get(),
            statistic=self.symmetry_statistic.get(),
            half_width_px=float(self.symmetry_half_width.get()),
            half_height_px=float(self.symmetry_half_height.get()),
            interpolation_order=1,
            minimum_contributors=max(
                1,
                int(self.symmetry_min_contributors.get()),
            ),
            compute_backend=self.compute_backend.get(),
        )

    def build_symmetry_product(self):
        if self.results is None:
            messagebox.showinfo(
                "No corrected image",
                "Load/process an image before symmetry averaging.",
            )
            return

        try:
            self.symmetry_result = build_symmetry_average(
                image=self.results["corrected"],
                mask=self.results["mask"],
                center_x=float(self.center_x.get()),
                center_y=float(self.center_y.get()),
                fiber_angle_deg=float(self.fiber_angle.get()),
                settings=self.get_symmetry_settings(),
            )

            summary = self.symmetry_result.summary()

            text = (
                f"{summary['number_of_symmetry_members']} symmetry-related "
                f"views combined by {summary['statistic']}.\n"
                f"Output: {summary['shape'][1]} × {summary['shape'][0]} pixels.\n"
                f"Ideal random-noise SNR gain: "
                f"~{summary['ideal_random_noise_snr_gain']:.2f}×."
            )

            if summary["mean_pairwise_member_correlation"] is not None:
                text += (
                    f"\nMean pairwise symmetry correlation: "
                    f"{summary['mean_pairwise_member_correlation']:.4f}."
                )

            if summary["normalized_rms_asymmetry"] is not None:
                text += (
                    f"\nNormalized RMS asymmetry: "
                    f"{summary['normalized_rms_asymmetry']:.4g}."
                )

            self.symmetry_summary_var.set(text)
            self.view_var.set("Symmetrized")
            self.refresh_plot()
            self.status_var.set("Symmetry average complete.")

        except Exception as exc:
            messagebox.showerror(
                "Symmetry averaging error",
                str(exc),
            )

    def preview_symmetry(self):
        if self.symmetry_result is None:
            messagebox.showinfo(
                "No symmetry result",
                "Build a symmetry average first.",
            )
            return

        self.view_var.set("Symmetrized")
        self.refresh_plot()

    def preview_asymmetry(self):
        if self.symmetry_result is None:
            messagebox.showinfo(
                "No symmetry result",
                "Build a symmetry average first.",
            )
            return

        self.view_var.set("Asymmetry")
        self.refresh_plot()

    # ============================================================
    # PROCESSING SETTINGS
    # ============================================================

    def get_processing_settings(self):
        return ProcessingSettings(
            dark_enabled=self.dark_enabled.get(),
            flat_enabled=self.flat_enabled.get(),

            background_enabled=self.background_enabled.get(),
            background_scale=float(self.background_scale.get()),

            normalize_enabled=self.normalize_enabled.get(),
            monitor_value=float(self.monitor_value.get()),

            hot_pixels_enabled=self.hot_pixels_enabled.get(),
            hot_pixel_sigma=float(self.hot_pixel_sigma.get()),
            hot_pixel_size=int(self.hot_pixel_size.get()),

            saturation_enabled=self.saturation_enabled.get(),
            saturation_value=float(self.saturation_value.get()),

            beamstop_enabled=self.beamstop_enabled.get(),
            beamstop_center_x=float(self.beamstop_center_x.get()),
            beamstop_center_y=float(self.beamstop_center_y.get()),
            beamstop_radius=float(self.beamstop_radius.get()),

            median_filter_enabled=self.median_filter_enabled.get(),
            median_filter_size=int(self.median_filter_size.get()),

            gaussian_filter_enabled=self.gaussian_filter_enabled.get(),
            gaussian_filter_sigma=float(self.gaussian_filter_sigma.get()),

            gaussian_background_enabled=self.gaussian_background_enabled.get(),
            gaussian_background_sigma=float(self.gaussian_background_sigma.get()),

            median_background_enabled=self.median_background_enabled.get(),
            median_background_size=int(self.median_background_size.get()),

            high_pass_enabled=self.high_pass_enabled.get(),
            high_pass_sigma=float(self.high_pass_sigma.get()),

            unsharp_enabled=self.unsharp_enabled.get(),
            unsharp_sigma=float(self.unsharp_sigma.get()),
            unsharp_amount=float(self.unsharp_amount.get()),

            display_mode=self.display_mode.get(),
            log_gain=float(self.log_gain.get()),
            gamma=float(self.gamma.get()),
            asinh_strength=float(self.asinh_strength.get()),
            custom_curve_spec=self.custom_curve_spec.get(),
            contrast_mode=self.contrast_mode.get(),
            percentile_low=float(self.percentile_low.get()),
            percentile_high=float(self.percentile_high.get()),
            manual_black=float(self.manual_black.get()),
            manual_white=float(self.manual_white.get()),
            robust_sigma=float(self.robust_sigma.get()),
            invert_display=self.invert_display.get(),
            local_contrast_enabled=self.local_contrast_enabled.get(),
            local_contrast_sigma=float(self.local_contrast_sigma.get()),
            local_contrast_strength=float(self.local_contrast_strength.get()),
            local_contrast_noise_floor=float(self.local_contrast_noise_floor.get()),
            histogram_bins=int(self.histogram_bins.get()),
            compute_backend=self.compute_backend.get(),
            fast_percentiles=self.fast_percentiles.get(),
            percentile_sample_target=int(self.percentile_sample_target.get()),
            fast_gaussian_background=self.fast_gaussian_background.get(),
            gaussian_background_downsample=int(self.gaussian_background_downsample.get()),
            fast_median_background=self.fast_median_background.get(),
            median_background_downsample=int(self.median_background_downsample.get()),
        )

    def reprocess(self, show_errors=True):
        if self.raw is None:
            return

        try:
            settings = self.get_processing_settings()

            self.results = self.processing_engine.process(
                self.raw,
                settings,
                dark=self.dark,
                flat=self.flat,
                background=self.background,
            )

            stage = self.results.get("_recomputed_stage", "cached")
            if stage == "quantitative":
                self.symmetry_result = None
                self.symmetry_summary_var.set(
                    "No symmetry average has been built."
                )

            backend = resolve_backend(settings.compute_backend)
            self.status_var.set(f"Processing ready — {backend.upper()} — recomputed: {stage}.")
            self.refresh_plot()

        except Exception as exc:
            self.status_var.set(
                f"Processing error: {exc}"
            )

            if show_errors:
                messagebox.showerror(
                    "Processing error",
                    str(exc),
                )

    def _bind_variable_updates(self):
        vars_to_watch = [
            self.dark_enabled,
            self.flat_enabled,
            self.background_enabled,
            self.normalize_enabled,
            self.hot_pixels_enabled,
            self.saturation_enabled,
            self.beamstop_enabled,

            self.median_filter_enabled,
            self.gaussian_filter_enabled,
            self.gaussian_background_enabled,
            self.median_background_enabled,
            self.high_pass_enabled,
            self.unsharp_enabled,

            self.display_mode,
            self.contrast_mode,
            self.invert_display,
            self.local_contrast_enabled,
            self.compute_backend,
            self.fast_percentiles,
            self.fast_gaussian_background,
            self.fast_median_background,
        ]

        for var in vars_to_watch:
            var.trace_add(
                "write",
                lambda *_args: self.reprocess(show_errors=False),
            )

        self.bind(
            "<Return>",
            lambda _e: self.reprocess(show_errors=False),
        )

    # ============================================================
    # IMAGE DISPLAY
    # ============================================================

    def refresh_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("black")

        selected_view = self.view_var.get()

        if selected_view == "Comparison":
            if self.comparison_result is None:
                self.ax.set_title("No comparison has been built.")
                self.canvas.draw_idle()
                return
            image = self.comparison_result.image
            title = (
                f"Comparison — {self.comparison_result.mode} — "
                f"scale B→A {self.comparison_result.scale_b_to_a:.5g}"
            )

        elif selected_view == "Stack":
            if self.stack_result is None:
                self.ax.set_title("No stack has been built.")
                self.canvas.draw_idle()
                return

            image = self.stack_result.image
            title = (
                f"Stack — {len(self.stack_result.frame_stats)} frames — "
                f"{self.stack_result.settings.method}"
            )

        elif selected_view in {
            "Symmetrized",
            "Asymmetry",
            "Symmetry contributors",
        }:
            if self.symmetry_result is None:
                self.ax.set_title("No symmetry average has been built.")
                self.canvas.draw_idle()
                return

            if selected_view == "Symmetrized":
                image = self.symmetry_result.symmetrized
                title = "Symmetrized corrected XRFD pattern"
            elif selected_view == "Asymmetry":
                image = self.symmetry_result.asymmetry_std
                title = "Quadrant disagreement: per-pixel standard deviation"
            else:
                image = self.symmetry_result.contributors
                title = "Number of symmetry contributors per pixel"

        else:
            if self.raw is None or self.results is None:
                self.ax.set_title(
                    "Open a TIFF or a folder of accumulations to begin."
                )
                self.canvas.draw_idle()
                return

            key_map = {
                "Raw": "raw",
                "Corrected": "corrected",
                "Enhanced": "enhanced",
                "Display": "display",
                "Mask": "mask",
                "Background model": "background_model",
            }

            key = key_map[self.view_var.get()]
            image = self.results.get(key)

            if image is None:
                self.ax.text(
                    0.5,
                    0.5,
                    "No background model is active.",
                    ha="center",
                    va="center",
                    transform=self.ax.transAxes,
                )
                self.canvas.draw_idle()
                return

            if key == "mask":
                image = image.astype(float)

            title = (
                f"{self.current_label or 'Image'} — "
                f"{self.view_var.get()}"
            )

        image = np.asarray(image)
        h_img, w_img = image.shape[:2]
        max_dim = max(256, int(self.viewer_max_dimension.get()))
        step = max(1, int(np.ceil(max(h_img, w_img) / max_dim)))
        shown = image[::step, ::step]

        self.ax.imshow(
            shown,
            cmap=self.cmap_var.get(),
            origin="upper",
            interpolation="nearest",
            extent=(0, w_img, h_img, 0),
        )

        self.ax.set_title(title)
        self.ax.set_xlabel("Detector X (pixels)")
        self.ax.set_ylabel("Detector Y (pixels)")

        # Fiber overlays only on current working-image views.
        if (
            self.raw is not None
            and self.view_var.get() not in {
                "Stack",
                "Symmetrized",
                "Asymmetry",
                "Symmetry contributors",
                "Comparison",
            }
        ):
            cx = self.center_x.get()
            cy = self.center_y.get()

            self.ax.plot(
                cx,
                cy,
                marker="+",
                markersize=12,
            )

            angle = np.deg2rad(self.fiber_angle.get())
            length = max(self.raw.shape) * 0.65

            dx = np.cos(angle) * length
            dy = np.sin(angle) * length

            self.ax.plot(
                [cx - dx, cx + dx],
                [cy - dy, cy + dy],
                linewidth=0.8,
                alpha=0.7,
            )

            ex = np.cos(angle - np.pi / 2) * length
            ey = np.sin(angle - np.pi / 2) * length

            self.ax.plot(
                [cx - ex, cx + ex],
                [cy - ey, cy + ey],
                linewidth=0.8,
                alpha=0.5,
            )

        # Non-destructive ROI / annotation overlays.
        if self.rois and selected_view not in {"Symmetrized", "Asymmetry", "Symmetry contributors"}:
            from matplotlib.patches import Circle, Ellipse, Rectangle
            for roi in self.rois:
                if not roi.visible:
                    continue
                kind = roi.shape_type.lower()
                if kind == "circle":
                    radius = float(np.hypot(roi.x1-roi.x0, roi.y1-roi.y0))
                    patch = Circle((roi.x0, roi.y0), radius, fill=False, linewidth=1.1)
                    self.ax.add_patch(patch)
                elif kind == "ellipse":
                    cx=0.5*(roi.x0+roi.x1); cy=0.5*(roi.y0+roi.y1)
                    patch=Ellipse((cx,cy), abs(roi.x1-roi.x0), abs(roi.y1-roi.y0), fill=False, linewidth=1.1)
                    self.ax.add_patch(patch)
                elif kind == "rectangle":
                    x=min(roi.x0,roi.x1); y=min(roi.y0,roi.y1)
                    patch=Rectangle((x,y),abs(roi.x1-roi.x0),abs(roi.y1-roi.y0),fill=False,linewidth=1.1)
                    self.ax.add_patch(patch)
                elif kind == "line":
                    self.ax.plot([roi.x0,roi.x1],[roi.y0,roi.y1],linewidth=1.1)
                if roi.label:
                    self.ax.text(roi.x0, roi.y0, roi.label, fontsize=8)

        # Layer-line overlays are analysis annotations in native detector coordinates.
        if self.layer_line_result is not None and selected_view not in {"Symmetrized", "Asymmetry", "Symmetry contributors", "Comparison"}:
            length = max(self.raw.shape) if self.raw is not None else max(image.shape)
            for line in self.layer_line_result.lines:
                offset = line.measured_offset_px if line.measured_offset_px is not None else line.offset_px
                xs, ys = fiber_to_detector(
                    np.array([offset, offset]),
                    np.array([-length, length]),
                    self.center_x.get(), self.center_y.get(), self.fiber_angle.get(),
                )
                self.ax.plot(xs, ys, linewidth=0.65, alpha=0.75)
                if abs(offset) < length:
                    tx, ty = fiber_to_detector(offset, 0.0, self.center_x.get(), self.center_y.get(), self.fiber_angle.get())
                    self.ax.text(float(tx)+4, float(ty)+4, f"L{line.index}", fontsize=7)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ============================================================
    # CONTRAST UTILITIES
    # ============================================================

    def _normalize_tone_curve_points(self, points):
        cleaned = []
        for x, y in points:
            if np.isfinite(x) and np.isfinite(y):
                cleaned.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))))
        if not cleaned:
            cleaned = [(0.0, 0.0), (1.0, 1.0)]
        cleaned.sort(key=lambda p: p[0])
        dedup = []
        for p in cleaned:
            if dedup and abs(p[0] - dedup[-1][0]) < 1e-5:
                dedup[-1] = p
            else:
                dedup.append(p)
        if dedup[0][0] > 1e-6:
            dedup.insert(0, (0.0, dedup[0][1]))
        else:
            dedup[0] = (0.0, dedup[0][1])
        if dedup[-1][0] < 1.0 - 1e-6:
            dedup.append((1.0, dedup[-1][1]))
        else:
            dedup[-1] = (1.0, dedup[-1][1])

        if self.tone_curve_monotonic.get():
            ys = np.maximum.accumulate([p[1] for p in dedup])
            dedup = [(p[0], float(np.clip(y, 0.0, 1.0))) for p, y in zip(dedup, ys)]
        return dedup

    def _tone_curve_to_spec(self):
        return ';'.join(f'{x:.7g},{y:.7g}' for x, y in self.tone_curve_points)

    def _sync_tone_curve(self, reprocess=True):
        self.tone_curve_points = self._normalize_tone_curve_points(self.tone_curve_points)
        self.custom_curve_spec.set(self._tone_curve_to_spec())
        self._draw_tone_curve()
        if reprocess and self.display_mode.get() == 'custom curve':
            self.reprocess(show_errors=False)

    def _draw_tone_curve(self):
        if not hasattr(self, 'tone_curve_ax'):
            return
        ax = self.tone_curve_ax
        ax.clear()
        ax.plot([0, 1], [0, 1], linestyle='--', linewidth=0.8, alpha=0.5, label='linear')
        pts = self.tone_curve_points
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker='o', linewidth=1.5, label='custom')
        if self.tone_curve_selected_index is not None and 0 <= self.tone_curve_selected_index < len(pts):
            p = pts[self.tone_curve_selected_index]
            ax.plot([p[0]], [p[1]], marker='s', markersize=8, linestyle='none')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Input intensity')
        ax.set_ylabel('Display output')
        ax.grid(True, alpha=0.18)
        self.tone_curve_fig.tight_layout(pad=0.6)
        self.tone_curve_canvas.draw_idle()

    def _nearest_tone_point(self, x, y, threshold=0.06):
        if x is None or y is None or not self.tone_curve_points:
            return None
        distances = [np.hypot(x - px, y - py) for px, py in self.tone_curve_points]
        idx = int(np.argmin(distances))
        return idx if distances[idx] <= threshold else None

    def _tone_curve_press(self, event):
        if event.inaxes != getattr(self, 'tone_curve_ax', None) or event.xdata is None or event.ydata is None:
            return
        x = float(np.clip(event.xdata, 0.0, 1.0))
        y = float(np.clip(event.ydata, 0.0, 1.0))
        idx = self._nearest_tone_point(x, y)

        if event.button == 3:
            if idx is not None and idx not in (0, len(self.tone_curve_points) - 1):
                self.tone_curve_points.pop(idx)
                self.tone_curve_selected_index = None
                self._sync_tone_curve(reprocess=True)
            return

        if event.button != 1:
            return

        if idx is None:
            self.tone_curve_points.append((x, y))
            self.tone_curve_points = self._normalize_tone_curve_points(self.tone_curve_points)
            idx = self._nearest_tone_point(x, y, threshold=0.12)

        self.tone_curve_selected_index = idx
        self.tone_curve_drag_index = idx
        self._draw_tone_curve()

    def _tone_curve_motion(self, event):
        idx = self.tone_curve_drag_index
        if idx is None or event.inaxes != getattr(self, 'tone_curve_ax', None) or event.xdata is None or event.ydata is None:
            return
        pts = list(self.tone_curve_points)
        x = float(np.clip(event.xdata, 0.0, 1.0))
        y = float(np.clip(event.ydata, 0.0, 1.0))

        if idx == 0:
            x = 0.0
        elif idx == len(pts) - 1:
            x = 1.0
        else:
            x = float(np.clip(x, pts[idx - 1][0] + 0.002, pts[idx + 1][0] - 0.002))

        if self.tone_curve_monotonic.get():
            lower = pts[idx - 1][1] if idx > 0 else 0.0
            upper = pts[idx + 1][1] if idx < len(pts) - 1 else 1.0
            y = float(np.clip(y, lower, upper))

        pts[idx] = (x, y)
        self.tone_curve_points = pts
        self._draw_tone_curve()

    def _tone_curve_release(self, _event):
        if self.tone_curve_drag_index is None:
            return
        self.tone_curve_drag_index = None
        self._sync_tone_curve(reprocess=True)

    def use_custom_tone_curve(self):
        self.display_mode.set('custom curve')
        self._sync_tone_curve(reprocess=True)

    def _set_tone_curve_points(self, points, activate=True):
        self.tone_curve_points = self._normalize_tone_curve_points(points)
        self.tone_curve_selected_index = None
        if activate:
            self.display_mode.set('custom curve')
        self._sync_tone_curve(reprocess=True)

    def tone_curve_linear(self):
        self._set_tone_curve_points([(0.0, 0.0), (1.0, 1.0)])

    def tone_curve_s_curve(self):
        self._set_tone_curve_points([
            (0.0, 0.0), (0.18, 0.08), (0.42, 0.34),
            (0.62, 0.72), (0.84, 0.94), (1.0, 1.0),
        ])

    def tone_curve_lift_shadows(self):
        self._set_tone_curve_points([
            (0.0, 0.0), (0.05, 0.16), (0.15, 0.31),
            (0.35, 0.52), (0.70, 0.82), (1.0, 1.0),
        ])

    def tone_curve_compress_highlights(self):
        self._set_tone_curve_points([
            (0.0, 0.0), (0.12, 0.20), (0.35, 0.48),
            (0.65, 0.72), (0.88, 0.88), (1.0, 0.94),
        ])

    def save_tone_curve(self):
        path = filedialog.asksaveasfilename(
            title='Save custom tone curve',
            defaultextension='.json',
            filetypes=[('JSON', '*.json')],
            initialfile='xrd_tone_curve.json',
        )
        if not path:
            return
        payload = {
            'format': 'xrd-image-toolkit-tone-curve-v1',
            'points': [[float(x), float(y)] for x, y in self.tone_curve_points],
            'monotonic': bool(self.tone_curve_monotonic.get()),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def load_tone_curve(self):
        path = filedialog.askopenfilename(
            title='Load custom tone curve',
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            points = payload.get('points', [])
            if len(points) < 2:
                raise ValueError('Tone-curve file must contain at least two points.')
            self.tone_curve_monotonic.set(bool(payload.get('monotonic', True)))
            self._set_tone_curve_points([(float(p[0]), float(p[1])) for p in points])
        except Exception as exc:
            messagebox.showerror('Tone curve error', str(exc))

    def apply_weak_reflection_preset(self):
        self.contrast_mode.set("source percentile")
        self.percentile_low.set(0.2)
        self.percentile_high.set(99.2)
        self.display_mode.set("asinh")
        self.asinh_strength.set(30.0)
        self.local_contrast_enabled.set(False)
        self.invert_display.set(False)
        self.reprocess(show_errors=False)

    def reset_contrast_defaults(self):
        self.display_mode.set("log")
        self.log_gain.set(100.0)
        self.gamma.set(0.5)
        self.asinh_strength.set(20.0)
        self.tone_curve_monotonic.set(True)
        self.tone_curve_points = [(0.0, 0.0), (1.0, 1.0)]
        self.custom_curve_spec.set("0,0;1,1")
        self._draw_tone_curve()
        self.contrast_mode.set("legacy percentile")
        self.percentile_low.set(0.5)
        self.percentile_high.set(99.7)
        self.manual_black.set(0.0)
        self.manual_white.set(1.0)
        self.robust_sigma.set(6.0)
        self.invert_display.set(False)
        self.local_contrast_enabled.set(False)
        self.local_contrast_sigma.set(25.0)
        self.local_contrast_strength.set(0.45)
        self.local_contrast_noise_floor.set(0.15)
        self.reprocess(show_errors=False)

    def show_contrast_histogram(self):
        if self.results is None:
            messagebox.showinfo("No image", "Load/process an image first.")
            return
        image = np.asarray(self.results["enhanced"], dtype=np.float32)
        step = max(1, int(np.sqrt(image.size / 300000)))
        values = image[::step, ::step].ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        win = tk.Toplevel(self)
        win.title("Intensity histogram / contrast levels")
        win.geometry("900x620")
        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(values, bins=min(512, max(64, int(np.sqrt(values.size)))), histtype="step")
        ax.set_yscale("log")
        ax.set_xlabel("Enhanced image intensity")
        ax.set_ylabel("Pixel count (log scale)")
        ax.set_title("Sampled intensity histogram")
        try:
            lo, hi = display_level_bounds(image, self.get_processing_settings())
            ax.axvline(lo, linestyle="--", label=f"black = {lo:.5g}")
            ax.axvline(hi, linestyle="--", label=f"white = {hi:.5g}")
            ax.legend()
        except Exception:
            pass
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # ============================================================
    # AUTOMATIC CENTER / FIBER-AXIS DETECTION
    # ============================================================

    def get_axis_detection_settings(self):
        return AxisDetectionSettings(
            analysis_max_dimension=max(200, int(self.axis_analysis_max_dimension.get())),
            center_search_radius_px=max(1.0, float(self.axis_center_search_radius.get())),
            center_refine_radius_px=max(0.5, float(self.axis_center_refine_radius.get())),
            center_refine_step_px=max(0.1, float(self.axis_center_refine_step.get())),
            coarse_angle_step_deg=max(0.25, float(self.axis_coarse_angle_step.get())),
            fine_angle_step_deg=max(0.02, float(self.axis_fine_angle_step.get())),
            central_exclusion_radius_px=max(0.0, float(self.axis_central_exclusion.get())),
        )

    def start_axis_detection_worker(self):
        if self.axis_worker_active:
            messagebox.showinfo("Detection running", "Automatic center/axis detection is already running.")
            return
        if self.results is None:
            messagebox.showinfo("No corrected image", "Load/process an image first.")
            return

        image = np.asarray(self.results["corrected"], dtype=np.float32).copy()
        mask = np.asarray(self.results["mask"], dtype=bool).copy()
        settings = self.get_axis_detection_settings()
        initial_x = float(self.center_x.get())
        initial_y = float(self.center_y.get())
        initial_angle = float(self.fiber_angle.get())

        self.axis_worker_active = True
        self.axis_detection_summary_var.set("Detecting center and symmetry axes…")
        self.status_var.set("Automatic center/axis detection running…")

        def worker():
            try:
                result = detect_beam_center_and_fiber_axis(
                    image,
                    mask=mask,
                    initial_center_x=initial_x,
                    initial_center_y=initial_y,
                    initial_fiber_angle_deg=initial_angle,
                    settings=settings,
                )
                self.axis_worker_queue.put(("done", result))
            except Exception:
                self.axis_worker_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_axis_worker)

    def _poll_axis_worker(self):
        try:
            while True:
                kind, payload = self.axis_worker_queue.get_nowait()
                if kind == "done":
                    self.axis_worker_active = False
                    self.axis_detection_result = payload
                    s = payload.summary()
                    self.axis_detection_summary_var.set(
                        f"Candidate center: X={s['center_x_px']:.2f}, Y={s['center_y_px']:.2f}\n"
                        f"Candidate fiber axis: {s['fiber_angle_deg']:.2f}°\n"
                        f"Center symmetry score: {s['center_symmetry_score']:.3f}\n"
                        f"Mirror-axis score: {s['mirror_symmetry_score']:.3f}"
                    )
                    self.status_var.set("Automatic center/axis candidate ready; review before applying.")
                elif kind == "error":
                    self.axis_worker_active = False
                    self.axis_detection_summary_var.set("Automatic detection failed.")
                    messagebox.showerror("Axis detection error", payload)
        except queue.Empty:
            pass
        if self.axis_worker_active:
            self.after(100, self._poll_axis_worker)

    def apply_axis_detection_result(self):
        if self.axis_detection_result is None:
            messagebox.showinfo("No candidate", "Run automatic center/axis detection first.")
            return
        self.center_x.set(self.axis_detection_result.center_x)
        self.center_y.set(self.axis_detection_result.center_y)
        self.fiber_angle.set(self.axis_detection_result.fiber_angle_deg)
        self.symmetry_result = None
        self.symmetry_summary_var.set("No symmetry average has been built.")
        self.refresh_plot()
        self.status_var.set("Detected center and fiber axis applied.")

    def show_axis_score_plot(self):
        if self.axis_detection_result is None:
            messagebox.showinfo("No candidate", "Run automatic center/axis detection first.")
            return
        result = self.axis_detection_result
        win = tk.Toplevel(self)
        win.title("Fiber-axis symmetry score")
        win.geometry("900x620")
        fig = Figure(figsize=(8.5, 5.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(result.coarse_angles_deg, result.coarse_scores, marker=".", label="coarse scan")
        ax.plot(result.fine_angles_deg, result.fine_scores, marker=".", label="fine scan")
        ax.axvline(result.fiber_angle_deg % 90.0, linestyle="--", label="selected symmetry axis")
        ax.set_xlabel("Axis angle modulo 90°")
        ax.set_ylabel("Mean mirror-symmetry correlation")
        ax.set_title("Automatic fiber-axis detection diagnostic")
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # ============================================================
    # INTERACTIVE IMAGE CLICKS
    # ============================================================

    def start_beam_center_click(self):
        self.click_mode = "beamstop_center"
        self.status_var.set(
            "Click the beam / beamstop center in the image."
        )

    def start_analysis_center_click(self):
        self.click_mode = "analysis_center"
        self.status_var.set(
            "Click the beam center in the image."
        )

    def start_axis_clicks(self):
        self.click_mode = "fiber_axis"
        self.axis_click_points = []
        self.status_var.set(
            "Click two points along the fiber axis."
        )

    def _on_image_click(self, event):
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return

        x = float(event.xdata)
        y = float(event.ydata)

        if self.roi_draw_mode is not None:
            self.roi_drag_start = (x, y)
            return

        if self.click_mode == "beamstop_center":
            self.beamstop_center_x.set(x)
            self.beamstop_center_y.set(y)
            self.click_mode = None
            self.reprocess(show_errors=False)

            self.status_var.set(
                f"Beamstop center set to ({x:.1f}, {y:.1f})."
            )

        elif self.click_mode == "analysis_center":
            self.center_x.set(x)
            self.center_y.set(y)
            self.click_mode = None
            self.refresh_plot()

            self.status_var.set(
                f"Beam center set to ({x:.1f}, {y:.1f})."
            )

        elif self.click_mode == "layer_anchor":
            parallel, _perp = detector_to_fiber(
                x, y, self.center_x.get(), self.center_y.get(), self.fiber_angle.get()
            )
            self.layer_line_anchor_offset.set(float(parallel))
            self.click_mode = None
            self.status_var.set(f"Layer-line anchor offset set to {float(parallel):.2f} px.")

        elif self.click_mode == "fiber_axis":
            self.axis_click_points.append((x, y))

            if len(self.axis_click_points) == 1:
                self.status_var.set(
                    "First axis point set. Click the second point."
                )

            elif len(self.axis_click_points) >= 2:
                (x1, y1), (x2, y2) = self.axis_click_points[:2]

                angle = np.degrees(
                    np.arctan2(y2 - y1, x2 - x1)
                )

                self.fiber_angle.set(angle)
                self.click_mode = None
                self.axis_click_points = []

                self.refresh_plot()

                self.status_var.set(
                    f"Fiber-axis angle set to {angle:.2f}°."
                )

    def _on_image_motion(self, event):
        if self.roi_draw_mode is None or self.roi_drag_start is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        # Lightweight status preview; final patch is created on mouse release.
        x0, y0 = self.roi_drag_start
        self.status_var.set(
            f"Drawing {self.roi_draw_mode}: ({x0:.1f},{y0:.1f}) → ({event.xdata:.1f},{event.ydata:.1f})"
        )

    def _on_image_release(self, event):
        if self.roi_draw_mode is None or self.roi_drag_start is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.roi_drag_start = None
            return
        x0, y0 = self.roi_drag_start
        x1, y1 = float(event.xdata), float(event.ydata)
        if np.hypot(x1-x0, y1-y0) < 2.0:
            self.roi_drag_start = None
            self.status_var.set("ROI was too small; drag farther to create it.")
            return
        self.roi_counter += 1
        roi = ROIShape(
            roi_id=f"ROI-{self.roi_counter:03d}",
            shape_type=self.roi_draw_mode,
            x0=float(x0), y0=float(y0), x1=x1, y1=y1,
            label=f"ROI-{self.roi_counter:03d}",
        )
        self.rois.append(roi)
        self.roi_drag_start = None
        self.roi_draw_mode = None
        self._refresh_roi_list()
        self.refresh_plot()
        self.status_var.set(f"Created {roi.shape_type} {roi.roi_id}.")

    def use_image_center(self):
        if self.raw is None:
            return

        h, w = self.raw.shape
        self.center_x.set((w - 1) / 2)
        self.center_y.set((h - 1) / 2)
        self.refresh_plot()

    # ============================================================
    # FIBER PROFILES
    # ============================================================

    def _get_analysis_image(self):
        if (
            self.use_symmetry_for_profiles.get()
            and self.symmetry_result is not None
        ):
            return self.symmetry_result.symmetrized

        if self.results is None:
            raise ValueError("No working image is loaded.")

        return self.results["corrected"]

    def _profiles(self):
        image = self._get_analysis_image()

        if (
            self.use_symmetry_for_profiles.get()
            and self.symmetry_result is not None
        ):
            h, w = image.shape
            center_x = (w - 1) / 2
            center_y = (h - 1) / 2
            fiber_angle = 90.0
        else:
            center_x = self.center_x.get()
            center_y = self.center_y.get()
            fiber_angle = self.fiber_angle.get()

        return extract_meridian_equator(
            image,
            center_x,
            center_y,
            fiber_angle_deg=fiber_angle,
            strip_width=max(1, int(self.strip_width.get())),
        )

    def plot_profiles(self):
        try:
            profiles = self._profiles()
            self._show_profile_window(
                profiles,
                show_peaks=False,
            )
        except Exception as exc:
            messagebox.showerror(
                "Profile error",
                str(exc),
            )

    def plot_profiles_with_peaks(self):
        try:
            profiles = self._profiles()
            self._show_profile_window(
                profiles,
                show_peaks=True,
            )
        except Exception as exc:
            messagebox.showerror(
                "Profile error",
                str(exc),
            )

    def _show_profile_window(self, profiles, show_peaks=False):
        win = tk.Toplevel(self)
        win.title("Fiber profiles")
        win.geometry("950x700")

        fig = Figure(
            figsize=(9, 6),
            dpi=100,
        )
        ax = fig.add_subplot(111)

        for label, (distance, intensity) in profiles.items():
            ax.plot(
                distance,
                intensity,
                label=label.capitalize(),
            )

            if show_peaks:
                prominence = self.peak_prominence.get()
                distance_px = self.peak_distance.get()

                peaks, _props = find_profile_peaks(
                    distance,
                    intensity,
                    prominence=(
                        prominence
                        if prominence > 0
                        else None
                    ),
                    distance_pixels=(
                        distance_px
                        if distance_px > 0
                        else None
                    ),
                )

                if len(peaks):
                    ax.plot(
                        distance[peaks],
                        intensity[peaks],
                        linestyle="none",
                        marker="o",
                        label=f"{label} peaks",
                    )

        ax.axvline(
            0,
            linewidth=0.8,
            alpha=0.4,
        )

        ax.set_xlabel(
            "Distance from beam center (pixels)"
        )
        ax.set_ylabel(
            "Corrected intensity"
        )
        ax.set_title(
            "Meridional and equatorial profiles"
        )
        ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(
            fig,
            master=win,
        )
        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )
        canvas.draw()

    def export_profiles(self):
        try:
            profiles = self._profiles()
        except Exception as exc:
            messagebox.showerror(
                "Profile error",
                str(exc),
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save fiber profiles",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=(
                f"{self.current_label}_fiber_profiles.csv"
                if self.current_label
                else "fiber_profiles.csv"
            ),
        )

        if not path:
            return

        md, mi = profiles["meridian"]
        ed, ei = profiles["equator"]

        n = min(
            len(md),
            len(ed),
        )

        save_csv(
            path,
            [
                md[:n],
                mi[:n],
                ed[:n],
                ei[:n],
            ],
            (
                "meridian_distance_px,meridian_intensity,"
                "equator_distance_px,equator_intensity"
            ),
        )

        self.status_var.set(
            f"Saved {Path(path).name}"
        )

    # ============================================================
    # ANALYSIS SESSION
    # ============================================================

    def _session_payload(self):
        return {
            "toolkit_version": APP_TITLE,
            "folder_path": str(self.folder_path) if self.folder_path else None,
            "files": [
                {
                    "path": str(entry["path"]),
                    "included": bool(entry.get("included", True)),
                    "exposure": entry.get("exposure"),
                    "exposure_source": entry.get("exposure_source"),
                    "exposure_info": entry.get("exposure_info"),
                }
                for entry in self.folder_entries
            ],
            "stacking": {
                "method": self.stack_method.get(),
                "align": self.stack_align.get(),
                "max_shift": self.stack_max_shift.get(),
                "normalization_mode": self.stack_normalization_mode.get(),
                "reference_exposure_seconds": self.stack_reference_exposure.get(),
                "input_correction_state": self.input_correction_state.get(),
                "exposure_auto_policy": self.exposure_auto_policy.get(),
                "sigma_clip_threshold": self.stack_sigma.get(),
                "sigma_clip_iterations": self.stack_iterations.get(),
                "trim_fraction": self.stack_trim_fraction.get(),
                "winsor_fraction": self.stack_winsor_fraction.get(),
                "huber_delta": self.stack_huber_delta.get(),
                "huber_iterations": self.stack_huber_iterations.get(),
            },
            "fiber_geometry": {
                "center_x": self.center_x.get(),
                "center_y": self.center_y.get(),
                "fiber_angle_deg": self.fiber_angle.get(),
                "strip_width": self.strip_width.get(),
            },
            "contrast": {
                "display_mode": self.display_mode.get(),
                "contrast_mode": self.contrast_mode.get(),
                "percentile_low": self.percentile_low.get(),
                "percentile_high": self.percentile_high.get(),
                "tone_curve_points": [list(v) for v in self.tone_curve_points],
                "tone_curve_monotonic": self.tone_curve_monotonic.get(),
                "invert_display": self.invert_display.get(),
            },
            "layer_lines": {
                "count": self.layer_line_count.get(),
                "anchor_offset_px": self.layer_line_anchor_offset.get(),
                "refine_radius_px": self.layer_line_refine_radius.get(),
                "inner_perp_px": self.layer_line_inner_perp.get(),
                "outer_perp_px": self.layer_line_outer_perp.get(),
                "half_width_px": self.layer_line_half_width.get(),
                "fit": self.layer_line_result.to_dict() if self.layer_line_result is not None else None,
            },
            "rois": [roi.to_dict() for roi in self.rois],
        }

    def save_analysis_session(self):
        path = filedialog.asksaveasfilename(
            title="Save XRD Image Toolkit session",
            defaultextension=".json",
            filetypes=[("Toolkit session JSON", "*.json"), ("All files", "*.*")],
            initialfile=f"{self._safe_current_stem()}_xrd_session.json",
        )
        if not path:
            return
        try:
            save_session(path, self._session_payload())
            self.status_var.set(f"Saved analysis session: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Session save", str(exc))

    def load_analysis_session(self):
        path = filedialog.askopenfilename(
            title="Load XRD Image Toolkit session",
            filetypes=[("Toolkit session JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            record = load_session(path)
            stack = record.get("stacking", {})
            setters = [
                (self.stack_method, stack.get("method")),
                (self.stack_align, stack.get("align")),
                (self.stack_max_shift, stack.get("max_shift")),
                (self.stack_normalization_mode, stack.get("normalization_mode")),
                (self.stack_reference_exposure, stack.get("reference_exposure_seconds")),
                (self.input_correction_state, stack.get("input_correction_state")),
                (self.exposure_auto_policy, stack.get("exposure_auto_policy")),
                (self.stack_sigma, stack.get("sigma_clip_threshold")),
                (self.stack_iterations, stack.get("sigma_clip_iterations")),
                (self.stack_trim_fraction, stack.get("trim_fraction")),
                (self.stack_winsor_fraction, stack.get("winsor_fraction")),
                (self.stack_huber_delta, stack.get("huber_delta")),
                (self.stack_huber_iterations, stack.get("huber_iterations")),
            ]
            fiber = record.get("fiber_geometry", {})
            setters += [
                (self.center_x, fiber.get("center_x")),
                (self.center_y, fiber.get("center_y")),
                (self.fiber_angle, fiber.get("fiber_angle_deg")),
                (self.strip_width, fiber.get("strip_width")),
            ]
            contrast = record.get("contrast", {})
            setters += [
                (self.display_mode, contrast.get("display_mode")),
                (self.contrast_mode, contrast.get("contrast_mode")),
                (self.percentile_low, contrast.get("percentile_low")),
                (self.percentile_high, contrast.get("percentile_high")),
                (self.tone_curve_monotonic, contrast.get("tone_curve_monotonic")),
                (self.invert_display, contrast.get("invert_display")),
            ]
            layer = record.get("layer_lines", {})
            setters += [
                (self.layer_line_count, layer.get("count")),
                (self.layer_line_anchor_offset, layer.get("anchor_offset_px")),
                (self.layer_line_refine_radius, layer.get("refine_radius_px")),
                (self.layer_line_inner_perp, layer.get("inner_perp_px")),
                (self.layer_line_outer_perp, layer.get("outer_perp_px")),
                (self.layer_line_half_width, layer.get("half_width_px")),
            ]
            for var, value in setters:
                if value is not None:
                    var.set(value)
            if contrast.get("tone_curve_points"):
                self.tone_curve_points = [tuple(map(float, pair)) for pair in contrast["tone_curve_points"]]
                self._sync_tone_curve_spec()
                self._draw_tone_curve()

            self.rois = [ROIShape.from_dict(v) for v in record.get("rois", [])]
            self.roi_counter = max([int(r.roi_id.split("-")[-1]) for r in self.rois if r.roi_id.split("-")[-1].isdigit()] or [0])
            self._refresh_roi_list()

            # Restore folder selection/exposure metadata when paths are accessible.
            files = record.get("files", [])
            existing = [Path(item["path"]) for item in files if Path(item["path"]).exists()]
            if existing:
                self.folder_path = Path(record.get("folder_path")) if record.get("folder_path") else existing[0].parent
                by_path = {str(Path(item["path"])): item for item in files}
                self.folder_entries = []
                for file_path in existing:
                    entry = self._new_entry(file_path, included=bool(by_path[str(file_path)].get("included", True)))
                    item = by_path[str(file_path)]
                    if item.get("exposure_info"):
                        entry["exposure_info"] = item.get("exposure_info")
                    if item.get("exposure") is not None:
                        entry["exposure"] = float(item["exposure"])
                        entry["exposure_source"] = item.get("exposure_source") or "session"
                    self.folder_entries.append(entry)
                valid_shapes = [e["shape"] for e in self.folder_entries if e.get("shape") is not None]
                if valid_shapes:
                    from collections import Counter
                    self.folder_common_shape = Counter(valid_shapes).most_common(1)[0][0]
                self.folder_label.configure(text=f"Session folder:\n{self.folder_path}\n{len(self.folder_entries)} accessible TIFF(s)")
                self._rebuild_file_tree()
                self._load_main_image(existing[0], keep_folder=True)
            else:
                self.refresh_plot()

            self.status_var.set(f"Loaded analysis session: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Session load", str(exc))

    # ============================================================
    # EXPORT
    # ============================================================

    def _current_view_image_and_title(self):
        """Return a display-ready version of the currently selected viewer image."""
        selected = self.view_var.get()

        if selected == "Comparison":
            if self.comparison_result is None:
                raise ValueError("No comparison has been built.")
            from .processing import display_transform
            rendered = display_transform(
                self.comparison_result.image,
                self.get_processing_settings(),
            )
            title = (
                f"Comparison — {self.comparison_result.mode} — "
                f"scale B→A {self.comparison_result.scale_b_to_a:.5g}"
            )
            return rendered, title

        if selected == "Stack":
            if self.stack_result is None:
                raise ValueError("No stack has been built.")

            from .processing import display_transform

            rendered = display_transform(
                self.stack_result.image,
                self.get_processing_settings(),
            )

            title = (
                f"Stack — {len(self.stack_result.frame_stats)} frames — "
                f"{self.stack_result.settings.method}"
            )
            return rendered, title

        if selected in {
            "Symmetrized",
            "Asymmetry",
            "Symmetry contributors",
        }:
            if self.symmetry_result is None:
                raise ValueError("No symmetry average has been built.")

            from .processing import display_transform

            if selected == "Symmetrized":
                rendered = display_transform(
                    self.symmetry_result.symmetrized,
                    self.get_processing_settings(),
                )
                title = "Symmetrized corrected XRFD pattern"
            elif selected == "Asymmetry":
                rendered = display_transform(
                    self.symmetry_result.asymmetry_std,
                    self.get_processing_settings(),
                )
                title = "Quadrant disagreement / asymmetry"
            else:
                rendered = self.symmetry_result.contributors.astype(float)
                title = "Number of symmetry contributors"

            return rendered, title

        if self.results is None:
            raise ValueError("No working image is loaded.")

        key_map = {
            "Raw": "raw",
            "Corrected": "corrected",
            "Enhanced": "enhanced",
            "Display": "display",
            "Mask": "mask",
            "Background model": "background_model",
        }

        key = key_map[selected]
        image = self.results.get(key)

        if image is None:
            raise ValueError(
                "The selected view does not currently contain image data."
            )

        title = f"{self.current_label or 'Image'} — {selected}"

        if key == "display":
            rendered = image
        elif key == "mask":
            rendered = image.astype(float)
        else:
            from .processing import display_transform
            rendered = display_transform(
                image,
                self.get_processing_settings(),
            )

        return rendered, title

    def _save_rendered_png(self, image, title, path):
        dpi = max(50, int(self.png_dpi.get()))

        fig = Figure(
            figsize=(8, 8),
            dpi=dpi,
        )
        ax = fig.add_subplot(111)

        ax.imshow(
            image,
            cmap=self.cmap_var.get(),
            origin="upper",
            interpolation="nearest",
        )

        if self.png_include_axes.get():
            ax.set_xlabel("Detector X (pixels)")
            ax.set_ylabel("Detector Y (pixels)")
        else:
            ax.set_axis_off()

        if self.png_include_title.get():
            ax.set_title(title)

        fig.tight_layout(
            pad=0.15 if not self.png_include_axes.get() else 1.0
        )

        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=(
                0.02 if not self.png_include_axes.get() else 0.1
            ),
        )

    def save_current_view_png(self):
        try:
            image, title = self._current_view_image_and_title()
        except Exception as exc:
            messagebox.showinfo(
                "Nothing to export",
                str(exc),
            )
            return

        selected = self.view_var.get().lower().replace(" ", "_")

        if self.view_var.get() == "Stack":
            base = (
                f"stack_{self.stack_result.settings.method.replace(' ', '_')}_"
                f"{len(self.stack_result.frame_stats)}frames"
            )
        else:
            base = self._safe_current_stem()

        path = filedialog.asksaveasfilename(
            title="Save current viewer as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"{base}_{selected}.png",
        )

        if not path:
            return

        self._save_rendered_png(
            image,
            title,
            path,
        )

        self.status_var.set(
            f"Saved PNG: {Path(path).name}"
        )

    def save_stack_png(self):
        if self.stack_result is None:
            messagebox.showinfo(
                "No stack",
                "Build a stack first.",
            )
            return

        from .processing import display_transform

        image = display_transform(
            self.stack_result.image,
            self.get_processing_settings(),
        )

        title = (
            f"Stack — {len(self.stack_result.frame_stats)} frames — "
            f"{self.stack_result.settings.method}"
        )

        path = filedialog.asksaveasfilename(
            title="Save stacked image as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=(
                f"stack_{self.stack_result.settings.method.replace(' ', '_')}_"
                f"{len(self.stack_result.frame_stats)}frames.png"
            ),
        )

        if not path:
            return

        self._save_rendered_png(
            image,
            title,
            path,
        )

        self.status_var.set(
            f"Saved stack PNG: {Path(path).name}"
        )

    def _ensure_results(self):
        if self.results is None:
            messagebox.showinfo(
                "Nothing to save",
                "Load or create a working image first.",
            )
            return False

        return True

    def _safe_current_stem(self):
        if self.current_path is not None:
            return self.current_path.stem

        if self.current_label:
            return (
                self.current_label
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
            )

        return "xrd_image"

    def _current_corrected_is_from_stack(self):
        return (
            self.current_source_kind == "stack"
            and self.stack_result is not None
        )

    def _detector_geometry_flags(self):
        stack_used = self._current_corrected_is_from_stack()
        registration_applied = bool(
            stack_used
            and getattr(self.stack_result.settings, "align", False)
        )

        # The corrected working image itself is never rotated/flipped/cropped/
        # interpolated/symmetry-folded by the processing pipeline. Temporal
        # registration is the one spatial exception because aligned stack
        # frames are shifted before combination.
        rotation_applied = False
        flip_applied = False
        crop_applied = False
        interpolation_applied = False
        symmetry_transform_applied = False

        detector_native = not any([
            rotation_applied,
            flip_applied,
            crop_applied,
            interpolation_applied,
            registration_applied,
            symmetry_transform_applied,
        ])

        return {
            "rotation_applied": rotation_applied,
            "flip_applied": flip_applied,
            "crop_applied": crop_applied,
            "interpolation_applied": interpolation_applied,
            "registration_applied": registration_applied,
            "symmetry_transform_applied": symmetry_transform_applied,
            "detector_native_orientation": detector_native,
        }

    def _corrected_provenance_payload(self, output_file=None):
        if self.results is None:
            raise ValueError("No corrected working image is available.")

        corrected = np.asarray(self.results["corrected"])
        settings = self.get_processing_settings()
        geometry = self._detector_geometry_flags()

        rows, columns = map(int, corrected.shape)

        expected_rows = int(self.provenance_expected_rows.get())
        expected_columns = int(self.provenance_expected_columns.get())

        shape_matches = (
            rows == expected_rows
            and columns == expected_columns
        )

        safe_for_poni = bool(
            shape_matches
            and geometry["detector_native_orientation"]
        )

        stack_used = self._current_corrected_is_from_stack()

        if stack_used:
            frame_paths = [
                stat.path
                for stat in self.stack_result.frame_stats
            ]
            stack_method = self.stack_result.settings.method
            frame_count = len(self.stack_result.frame_stats)
            stack_summary = self.stack_result.summary()
            approx_snr = stack_summary.get(
                "approximate_snr_gain_vs_one_frame"
            )
        else:
            frame_paths = []
            stack_method = None
            frame_count = 1 if self.current_source_kind == "tiff" else None
            approx_snr = None

        mask = np.asarray(self.results["mask"], dtype=bool)
        masked_count = int(np.count_nonzero(mask))

        output_path = Path(output_file) if output_file else None
        output_hash = (
            sha256_file(output_path)
            if output_path is not None and output_path.exists()
            else None
        )

        calibration_label = self.provenance_calibration_label.get().strip()

        compatibility_note = (
            "This detector-coordinate product is intended for the named calibration "
            "only if that PONI/geometry file was calibrated against the same native "
            "detector row/column orientation, detector dimensions, pixel size, and "
            "detector geometry. Recording the label does not verify the PONI contents."
        )

        if safe_for_poni:
            compatibility_status = "intended_unverified"
        else:
            compatibility_status = "geometry_not_native_or_shape_mismatch"

        return {
            "provenance": {
                "schema_name": "xrd_image_toolkit_provenance",
                "schema_version": "1.0",
                "generated_utc": utc_timestamp(),
                "toolkit": APP_TITLE,
            },
            "product": {
                "product_type": "corrected_detector_coordinate_image",
                "output_file": str(output_path) if output_path else None,
                "output_file_sha256": output_hash,
                "array_rows": rows,
                "array_columns": columns,
                "array_dtype_in_memory": str(corrected.dtype),
                "exported_tiff_dtype": "float32",
                "source_kind": self.current_source_kind,
                "source_file": (
                    str(self.current_path)
                    if self.current_path is not None
                    else None
                ),
                "source_label": self.current_label,
            },
            "coordinate_system": {
                "coordinate_system": (
                    "detector_native"
                    if geometry["detector_native_orientation"]
                    else "detector_modified"
                ),
                "axis_0": "detector_row",
                "axis_1": "detector_column",
                "array_origin": "upper_left",
                **geometry,
            },
            "intensity_processing": {
                "dark_subtraction_applied": bool(settings.dark_enabled),
                "flat_field_applied": bool(settings.flat_enabled),
                "background_subtraction_applied": bool(
                    settings.background_enabled
                ),
                "background_scale": float(settings.background_scale),
                "monitor_normalization_applied": bool(
                    settings.normalize_enabled
                ),
                "monitor_value": float(settings.monitor_value),
                "hot_pixel_mask_applied": bool(
                    settings.hot_pixels_enabled
                ),
                "saturation_mask_applied": bool(
                    settings.saturation_enabled
                ),
                "beamstop_mask_applied": bool(
                    settings.beamstop_enabled
                ),
                "masked_pixel_count": masked_count,
                "enhancement_filter_applied": False,
                "display_transform_applied": False,
                "display_intensity_rescaling_applied": False,
                "custom_tone_curve_applied": False,
            },
            "stacking": {
                "temporal_stack_used": bool(stack_used),
                "stack_method": stack_method,
                "frame_count": frame_count,
                "registration_enabled": bool(
                    geometry["registration_applied"]
                ),
                "input_files": frame_paths,
                "approximate_ideal_snr_gain": approx_snr,
                "input_correction_state": (
                    self.stack_result.settings.input_correction_state
                    if stack_used else self.input_correction_state.get()
                ),
                "frame_normalization_mode": (
                    self.stack_result.settings.normalization_mode
                    if stack_used else "none"
                ),
                "reference_exposure_seconds": (
                    self.stack_result.settings.reference_exposure_seconds
                    if stack_used else None
                ),
                "frame_exposures_seconds": (
                    {stat.filename: stat.exposure_seconds for stat in self.stack_result.frame_stats}
                    if stack_used else {}
                ),
                "frame_exposure_sources": (
                    {stat.filename: stat.exposure_source for stat in self.stack_result.frame_stats}
                    if stack_used else {}
                ),
                "frame_exposure_candidates": (
                    {
                        stat.filename: {
                            "filename_exposure_seconds": stat.filename_exposure_seconds,
                            "header_exposure_seconds": stat.header_exposure_seconds,
                            "header_integration_seconds": stat.header_integration_seconds,
                            "exposure_mismatch": stat.exposure_mismatch,
                        }
                        for stat in self.stack_result.frame_stats
                    }
                    if stack_used else {}
                ),
            },
            "reference_frames": {
                "dark_file": (
                    str(self.dark_path)
                    if self.dark_path
                    else None
                ),
                "flat_file": (
                    str(self.flat_path)
                    if self.flat_path
                    else None
                ),
                "background_file": (
                    str(self.background_path)
                    if self.background_path
                    else None
                ),
            },
            "fiber_geometry": {
                "beam_center_x_px": float(self.center_x.get()),
                "beam_center_y_px": float(self.center_y.get()),
                "fiber_angle_deg_from_positive_x": float(
                    self.fiber_angle.get()
                ),
                "strip_width_px": int(self.strip_width.get()),
            },
            "calibration": {
                "intended_geometry_calibration": (
                    calibration_label or None
                ),
                "compatibility_status": compatibility_status,
                "expected_detector_rows": expected_rows,
                "expected_detector_columns": expected_columns,
                "shape_matches_expected_detector": bool(shape_matches),
                "safe_detector_geometry_for_existing_poni": bool(
                    safe_for_poni
                ),
                "compatibility_note": compatibility_note,
            },
            "tiff_header": {
                "original_tiff_header_preserved": False,
                "export_header_generated_by_tifffile": True,
                "analysis_implication": (
                    "The toolkit writes a new TIFF container and does not copy "
                    "the original acquisition header/tags verbatim. Retain the "
                    "original TIFFs and this provenance record for acquisition "
                    "metadata needed in later analysis."
                ),
            },
            "analysis_guidance": {
                "recommended_for_poni_geometry": bool(safe_for_poni),
                "recommended_for_quantitative_integration": True,
                "use_symmetrized_product_with_original_poni": False,
                "use_png_for_quantitative_analysis": False,
                "note": (
                    "Use the corrected detector-coordinate TIFF for calibrated "
                    "integration/model comparison when the recorded geometry checks "
                    "are satisfied. Display PNGs and symmetrized products serve "
                    "different purposes and should not be substituted for this array."
                ),
            },
        }

    def _write_corrected_provenance_sidecars(self, corrected_tiff_path):
        corrected_tiff_path = Path(corrected_tiff_path)
        sidecar_stem = corrected_tiff_path.with_suffix("")
        sidecar_stem = sidecar_stem.with_name(
            sidecar_stem.name + ".provenance"
        )

        payload = self._corrected_provenance_payload(
            output_file=corrected_tiff_path
        )

        return write_provenance_bundle(
            sidecar_stem,
            payload,
        )

    def save_provenance_bundle(self):
        if not self._ensure_results():
            return

        path = filedialog.asksaveasfilename(
            title="Save corrected-image provenance bundle",
            defaultextension=".json",
            filetypes=[
                ("Provenance JSON", "*.json"),
                ("Text file", "*.txt"),
                ("All files", "*.*"),
            ],
            initialfile=f"{self._safe_current_stem()}_provenance.json",
        )

        if not path:
            return

        path = Path(path)
        stem = path.with_suffix("")

        try:
            payload = self._corrected_provenance_payload(
                output_file=None
            )
            json_path, txt_path = write_provenance_bundle(
                stem,
                payload,
            )

            self.status_var.set(
                f"Saved provenance: {json_path.name} + {txt_path.name}"
            )
        except Exception as exc:
            messagebox.showerror(
                "Provenance export error",
                str(exc),
            )

    def save_corrected(self):
        if not self._ensure_results():
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            initialfile=f"{self._safe_current_stem()}_corrected.tif",
        )

        if not path:
            return

        try:
            save_float_tiff(
                path,
                self.results["corrected"],
            )

            if self.provenance_auto_sidecars.get():
                json_path, txt_path = self._write_corrected_provenance_sidecars(
                    path
                )
                self.status_var.set(
                    f"Saved {Path(path).name} with provenance sidecars "
                    f"{json_path.name} and {txt_path.name}"
                )
            else:
                self.status_var.set(
                    f"Saved corrected TIFF: {Path(path).name}"
                )

        except Exception as exc:
            messagebox.showerror(
                "Corrected TIFF export error",
                str(exc),
            )

    def save_enhanced(self):
        if not self._ensure_results():
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            initialfile=f"{self._safe_current_stem()}_enhanced.tif",
        )

        if path:
            save_float_tiff(
                path,
                self.results["enhanced"],
            )

    def save_mask(self):
        if not self._ensure_results():
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            initialfile=f"{self._safe_current_stem()}_mask.tif",
        )

        if path:
            save_mask_tiff(
                path,
                self.results["mask"],
            )

    def save_display_png(self):
        if not self._ensure_results():
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=f"{self._safe_current_stem()}_display.png",
        )

        if not path:
            return

        self._save_rendered_png(
            self.results["display"],
            self.current_label or "XRD image",
            path,
        )

    def _metadata_payload(self):
        return {
            "toolkit": APP_TITLE,
            "source_file": (
                str(self.current_path)
                if self.current_path is not None
                else self.current_label
            ),
            "processing_settings": (
                self.get_processing_settings().to_dict()
            ),
            "reference_images": {
                "dark": (
                    str(self.dark_path)
                    if self.dark_path
                    else None
                ),
                "flat": (
                    str(self.flat_path)
                    if self.flat_path
                    else None
                ),
                "background": (
                    str(self.background_path)
                    if self.background_path
                    else None
                ),
            },
            "fiber_analysis": {
                "beam_center_x_px": self.center_x.get(),
                "beam_center_y_px": self.center_y.get(),
                "fiber_angle_deg_from_positive_x": self.fiber_angle.get(),
                "strip_width_px": self.strip_width.get(),
            },
        }

    def save_metadata(self):
        if not self._ensure_results():
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{self._safe_current_stem()}_processing.json",
        )

        if path:
            save_json(
                path,
                self._metadata_payload(),
            )

    def save_symmetry_tiff(self):
        if self.symmetry_result is None:
            messagebox.showinfo(
                "No symmetry result",
                "Build a symmetry average first.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save symmetrized corrected image",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            initialfile=f"{self._safe_current_stem()}_symmetrized.tif",
        )

        if path:
            save_float_tiff(
                path,
                self.symmetry_result.symmetrized,
            )

    def save_asymmetry_tiff(self):
        if self.symmetry_result is None:
            messagebox.showinfo(
                "No symmetry result",
                "Build a symmetry average first.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save symmetry disagreement map",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            initialfile=f"{self._safe_current_stem()}_asymmetry_std.tif",
        )

        if path:
            save_float_tiff(
                path,
                self.symmetry_result.asymmetry_std,
            )

    def save_symmetry_png(self):
        if self.symmetry_result is None:
            messagebox.showinfo(
                "No symmetry result",
                "Build a symmetry average first.",
            )
            return

        from .processing import display_transform

        rendered = display_transform(
            self.symmetry_result.symmetrized,
            self.get_processing_settings(),
        )

        path = filedialog.asksaveasfilename(
            title="Save symmetrized image as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"{self._safe_current_stem()}_symmetrized.png",
        )

        if not path:
            return

        self._save_rendered_png(
            rendered,
            "Symmetrized corrected XRFD pattern",
            path,
        )

    def save_symmetry_report(self):
        if self.symmetry_result is None:
            messagebox.showinfo(
                "No symmetry result",
                "Build a symmetry average first.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save symmetry report",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"{self._safe_current_stem()}_symmetry_report.json",
        )

        if not path:
            return

        save_json(
            path,
            {
                "toolkit": APP_TITLE,
                "summary": self.symmetry_result.summary(),
                "settings": self.symmetry_result.settings.to_dict(),
                "pairwise_member_correlations": (
                    self.symmetry_result.correlations.tolist()
                ),
            },
        )

    def save_stack_tiff(self):
        if self.stack_result is None:
            messagebox.showinfo(
                "No stack",
                "Build a stack first.",
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif")],
            initialfile=(
                f"stack_{self.stack_result.settings.method.replace(' ', '_')}_"
                f"{len(self.stack_result.frame_stats)}frames.tif"
            ),
        )

        if path:
            save_float_tiff(
                path,
                self.stack_result.image,
            )

    def save_stack_report(self):
        if self.stack_result is None:
            messagebox.showinfo(
                "No stack",
                "Build a stack first.",
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save stack report",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="stack_report.json",
        )

        if not path:
            return

        path = Path(path)

        payload = {
            "toolkit": APP_TITLE,
            "summary": self.stack_result.summary(),
            "settings": self.stack_result.settings.to_dict(),
            "frames": [
                stat.to_dict()
                for stat in self.stack_result.frame_stats
            ],
        }

        save_json(
            path,
            payload,
        )

        csv_path = path.with_name(
            f"{path.stem}_frames.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "path",
                    "filename",
                    "included",
                    "mean",
                    "median",
                    "maximum",
                    "correlation_to_reference",
                    "noise_sigma",
                    "stack_weight",
                    "shift_y",
                    "shift_x",
                    "exposure_seconds",
                    "exposure_source",
                    "filename_exposure_seconds",
                    "header_exposure_seconds",
                    "header_integration_seconds",
                    "exposure_mismatch",
                    "normalization_factor",
                ],
            )

            writer.writeheader()

            for stat in self.stack_result.frame_stats:
                writer.writerow(
                    stat.to_dict()
                )

        self.status_var.set(
            f"Saved stack report and {csv_path.name}"
        )


def main():
    app = XRDToolkitApp()
    app.mainloop()
