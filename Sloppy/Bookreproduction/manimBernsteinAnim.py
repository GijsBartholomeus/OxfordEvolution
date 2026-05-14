"""
Bernstein ellipse animation in Manim (Community Edition).

Shows: the strip  S_rho = { phi in C : |Im phi| < log rho }  deforms under
       z = cos(phi)  into the interior of the Bernstein ellipse E_rho.

3Blue1Brown-style grid deformation, rho = 2, ~10 seconds.

----- HOW TO RUN -----

In your bioevo mamba env (or any fresh one), install Manim from conda-forge --
it bundles pango, cairo, ffmpeg and a working LaTeX, so no system root needed:

    mamba install -n bioevo -c conda-forge manim
    # or, in a clean env:
    mamba create -n manim -c conda-forge python=3.11 manim
    mamba activate manim

Render the 10-second MP4 (1080p, 60 fps):

    manim -qh bernstein_manim.py BernsteinEllipse

Render a GIF as well:

    manim -qh --format=gif bernstein_manim.py BernsteinEllipse

Quality flags:
    -ql  low (480p15)   -- fast iteration
    -qm  medium (720p30)
    -qh  high   (1080p60) -- recommended for slides
    -qk  4K

Output lands in ./media/videos/bernstein_manim/<resolution>/.
"""

from pathlib import Path
import os
import sys

# Set pydub ffmpeg location before any imports that use it
ffmpeg_path = os.path.join(sys.prefix, "bin", "ffmpeg")
if os.path.exists(ffmpeg_path):
    os.environ["PYDUB_FFMPEG_LOCATION"] = ffmpeg_path

from manim import *
import numpy as np


config.video_dir = str(Path(__file__).resolve().parent)


# --- problem parameters -----------------------------------------------------
RHO = 2.0
L = np.log(RHO)              # strip half-width, log(2) ≈ 0.6931
A = np.cosh(L)               # semi-major axis of E_rho = (rho+1/rho)/2 = 1.25
B = np.sinh(L)               # semi-minor axis of E_rho = (rho-1/rho)/2 = 0.75

X_MIN, X_MAX = -PI, PI       # one period of cos in the real direction
Y_MIN, Y_MAX = -L, L         # the strip

N_HORIZ = 11                 # horizontal grid lines (constant Im phi)
N_VERT  = 17                 # vertical   grid lines (constant Re phi)
N_PTS   = 220                # sample points per line (smooth curves)

# --- colors -----------------------------------------------------------------
BG               = "#0b0d12"
TXT              = "#e9eef5"
COLOR_BOUNDARY   = "#ffd24d"   # y = ±L  ->  E_rho
COLOR_AXIS       = "#ff5c7a"   # y = 0   ->  segment [-1, 1]
COLOR_VERT       = "#5bb6e8"   # vertical lines
COLOR_INTERIOR_LO = "#c44ad9"
COLOR_INTERIOR_HI = "#f08a3c"


def interior_color(y: float):
    """Color for an interior horizontal line, blending by |y|/L."""
    t = abs(y) / L
    return interpolate_color(
        ManimColor(COLOR_INTERIOR_LO),
        ManimColor(COLOR_INTERIOR_HI),
        t,
    )


def horiz_curve(y: float, color, stroke_width: float):
    return ParametricFunction(
        lambda t, y=y: np.array([t, y, 0.0]),
        t_range=(X_MIN, X_MAX),
        color=color,
        stroke_width=stroke_width,
    )


def vert_curve(x: float, color, stroke_width: float):
    return ParametricFunction(
        lambda t, x=x: np.array([x, t, 0.0]),
        t_range=(Y_MIN, Y_MAX),
        color=color,
        stroke_width=stroke_width,
    )


class BernsteinEllipse(Scene):
    """Strip in the phi-plane is deformed by z = cos(phi) into E_rho."""

    def construct(self):
        self.camera.background_color = BG

        # ---- viewport: comfortably show both the strip and the ellipse
        plane = NumberPlane(
            x_range=(-3.8, 3.8, 1),
            y_range=(-2.0, 2.0, 1),
            x_length=14,
            y_length=7.5,
            background_line_style={
                "stroke_color": "#1a1f2b",
                "stroke_width": 1,
                "stroke_opacity": 0.6,
            },
            axis_config={"stroke_color": "#3a4252", "stroke_width": 1.2},
        )
        self.add(plane)

        # ---- title at the top
        title = MathTex(
            r"\{\,\varphi\in\mathbb{C}\;:\;|\mathrm{Im}\,\varphi|<\log\rho\,\}",
            r"\;\xrightarrow{\;z=\cos\varphi\;}\;",
            r"E_\rho",
            r"\quad(\rho=2)",
            font_size=36,
        ).set_color(TXT).to_edge(UP, buff=0.35)
        self.add(title)

        # ---- target ellipse drawn faintly from the start
        target_ellipse = Ellipse(
            width=2 * A, height=2 * B,
            color=COLOR_BOUNDARY, stroke_width=1.5, stroke_opacity=0.22,
        )
        self.add(target_ellipse)

        # ---- foci markers at ±1 (foci of every Bernstein ellipse)
        focus_dots = VGroup(
            Dot(np.array([-1, 0, 0]), radius=0.045, color="#9fb3c8"),
            Dot(np.array([ 1, 0, 0]), radius=0.045, color="#9fb3c8"),
        )
        focus_labels = VGroup(
            MathTex("-1", font_size=24, color="#9fb3c8").next_to(focus_dots[0], DOWN, buff=0.08),
            MathTex("+1", font_size=24, color="#9fb3c8").next_to(focus_dots[1], DOWN, buff=0.08),
        )
        self.add(focus_dots, focus_labels)

        # ---- build the grid in the strip ----
        y_vals = np.linspace(Y_MIN, Y_MAX, N_HORIZ)
        x_vals = np.linspace(X_MIN, X_MAX, N_VERT)

        horiz_lines = VGroup()
        for y in y_vals:
            if abs(abs(y) - L) < 1e-9:
                c, sw = COLOR_BOUNDARY, 5
            elif abs(y) < 1e-9:
                c, sw = COLOR_AXIS, 3.5
            else:
                c, sw = interior_color(y), 2
            horiz_lines.add(horiz_curve(y, c, sw))

        vert_lines = VGroup()
        for x in x_vals:
            vert_lines.add(vert_curve(x, COLOR_VERT, 1.5))

        grid = VGroup(horiz_lines, vert_lines)

        # caption at the bottom
        caption = Tex("strip in the $\\varphi$-plane", font_size=30, color=TXT)
        caption.to_edge(DOWN, buff=0.4)
        self.add(caption)

        # ---- 1. draw the grid (~1.2 s)
        self.play(
            LaggedStartMap(Create, horiz_lines, lag_ratio=0.04, run_time=1.2),
            LaggedStartMap(Create, vert_lines,  lag_ratio=0.03, run_time=1.2),
        )

        # ---- 2. hold briefly so the strip is readable (~0.8 s)
        self.wait(0.8)

        # ---- 3. morph under cos(phi) (~5.5 s)
        def cos_map(point):
            z = complex(point[0], point[1])
            w = np.cos(z)
            return np.array([w.real, w.imag, 0.0])

        new_caption = Tex("image is the interior of $E_2$", font_size=30, color=TXT).move_to(caption)

        self.play(
            grid.animate.apply_function(cos_map),
            target_ellipse.animate.set_stroke(opacity=0.85),
            FadeTransform(caption, new_caption),
            run_time=5.5,
            rate_func=smooth,
        )

        # ---- 4. hold on the ellipse (~2 s)
        self.wait(2.0)