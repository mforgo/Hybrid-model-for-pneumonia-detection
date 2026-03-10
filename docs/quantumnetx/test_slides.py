"""
Hybridní Kvantově-Klasický model – Presentation Slides
=======================================================
Run with:
    manim-slides render test_slides.py MyPresentation
    manim-slides present MyPresentation
"""

from manim import *
from manim_slides import Slide, ThreeDSlide
import numpy as np
import random

# ── Global constants ────────────────────────────────────────────────────────
SLIDE_RANGE = slice(0, 26)  # Change to slice(None) to render all slides

# Colour palette (used across multiple slides)
W_COLOR   = WHITE
A_COLOR   = YELLOW
B_COLOR   = BLUE
SIG_COLOR = GREEN

# Chart colours (slide_16)
COLOR_NORMAL = "#87CEFA"  # Light Sky Blue  – normal class bars
COLOR_PNEU   = "#FA8072"  # Salmon/Light Red – pneumonia class bars
COLOR_PIXEL_STROKE = "#FFFFFF"  # pixel grid stroke in slide_05
COLOR_GRID_BORDER  = "#59C1BD"  # teal border around pixel grid in slide_05

# ── Typographic scale ────────────────────────────────────────────────────────
# All font_size values in the file must come from this table so that the
# presentation has a consistent visual hierarchy.  Avoid raw integers in code.
FS_TINY    = 14   # fine-print / disclaimers
FS_CAPTION = 16   # image captions, minor sub-labels
FS_SMALL   = 20   # legend entries, axis tick labels, annotation details
FS_BODY    = 24   # body text, axis labels, code snippets, bar-chart numbers
FS_SUB     = 28   # sub-titles, secondary callouts
FS_BASE    = 32   # widget labels, circuit annotations
FS_HEADING = 36   # prominent formula text
FS_TITLE   = 40   # slide section headings (chart titles, formula blocks)
FS_SLIDE   = 44   # main slide title (most slides)
FS_LARGE   = 48   # emphasized slide titles / ResNet, architecture
FS_HERO    = 56   # title-slide headline text
FS_DISPLAY = 76   # very large display text (slide_03 word highlight)


# ── Image asset paths ────────────────────────────────────────────────────────
# All image references go through these constants.  Update here if the asset
# directory ever moves.
IMG_DIR       = "media/images/test_slides"
IMG_CXR       = f"{IMG_DIR}/cxr.png"
IMG_KAGGLE    = f"{IMG_DIR}/kaggle.png"
IMG_CIRCUIT   = f"{IMG_DIR}/circuit.png"
IMG_INFIS     = f"{IMG_DIR}/infis.png"
IMG_NTC       = f"{IMG_DIR}/ntc.png"
IMG_VLAJKY    = f"{IMG_DIR}/vlajky.png"
IMG_INTERREG  = f"{IMG_DIR}/interreg.png"
IMG_REPO_QR   = f"{IMG_DIR}/repo_qr.png"
IMG_DOCS_QR   = f"{IMG_DIR}/docs_qr.png"


# ── Výsledky modelu (slide_24) ───────────────────────────────────────────────
# ⬇️  SEM VLOŽTE SVÁ SKUTEČNÁ DATA Z NOTEBOOKU  ⬇️
# Nahraďte seznamy níže reálnou historií ztrát z tréninku.
# Délky TRAIN_LOSS a VAL_LOSS musí být stejné (počet epoch).
TRAIN_LOSS = [0.85, 0.72, 0.65, 0.58, 0.52, 0.48, 0.45, 0.42, 0.40, 0.38]
VAL_LOSS   = [0.84, 0.75, 0.68, 0.60, 0.55, 0.52, 0.50, 0.48, 0.45, 0.43]

# Finální klasifikační metriky na testovacím setu (hodnoty 0.0 – 1.0).
FINAL_ACCURACY    = 0.87
FINAL_SENSITIVITY = 0.92   # senzitivita – klíčová metrika pro medicínu
FINAL_SPECIFICITY = 0.78



def _weight_color(w: float) -> str:
    """Map a weight value to a colour: positive → blue, negative → red.
    The input is clamped to [-1, 1]; intensity scales with magnitude.
    """
    t = max(-1.0, min(1.0, w))
    if t >= 0:
        r = int(30  + (1 - t) * 60)
        g = int(80  + (1 - t) * 60)
        b = int(200 + t * 55)
    else:
        t = abs(t)
        r = int(200 + t * 55)
        g = int(80  - t * 60)
        b = int(30  + (1 - t) * 60)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _make_label(text: str, color: ManimColor, scale: float = 0.6) -> MathTex:
    return MathTex(rf"\text{{{text}}}", color=color).scale(scale)


def _fade_all(scene: Scene) -> None:
    """Fade every current mobject off-screen.
    Uses a single FadeOut call to avoid hitting the PyAV mux limit when
    many mobjects are on screen (e.g. large grids or NN edges).
    """
    mobs = list(scene.mobjects)
    if mobs:
        scene.play(FadeOut(*mobs), run_time=1.5)
    scene.wait(0.5)


# ── Presentation ────────────────────────────────────────────────────────────
class MyPresentation(ThreeDSlide):
    """
    Slides are defined as slide_NN methods so they are auto-discovered and
    executed in alphabetical order.  Adjust SLIDE_RANGE at the top of the
    file to preview a single slide during development.
    """

    def construct(self):
        methods = sorted(
            name for name in dir(self)
            if callable(getattr(self, name)) and name.startswith("slide_")
        )
        for name in methods[SLIDE_RANGE]:
            self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
            getattr(self, name)()
        # Closing slide always runs last, regardless of SLIDE_RANGE
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        self.last_slide()

    # ── slide_00: prázdný úvodní snímek ─────────────────────────────────────
    # Deliberate blank slide.  manim-slides starts presentation in a "paused"
    # state; this gives the presenter a clean black frame to hold on while
    # the audience settles, before advancing to the title slide.
    def slide_00(self):
        self.next_slide(notes="Úvodní prázdný snímek – počkejte na publikum.")

    # ── slide_01: title ─────────────────────────────────────────────────────
    def slide_01(self):
        title = VGroup(
            Text("Hybridní Kvantově-Klasický model", font_size=FS_HERO, weight=BOLD),
            Text("klasifikující CXR snímky plic",    font_size=FS_HERO, weight=BOLD),
            Text("s pneumonií",                       font_size=FS_HERO, weight=BOLD),
        ).arrange(DOWN, buff=0.2)

        author = Text("Michal Forgó", font_size=FS_HEADING)
        text_group = VGroup(title, author).arrange(DOWN, buff=0.7)
        text_group.move_to(ORIGIN + UP * 0.5)

        CORNER_BUFF  = 0.2   # gap from screen edge for corner logos
        CORNER_H     = 1.0   # height of corner logos (infis / ntc)
        INTERREG_H   = 1.2   # interreg must be strictly larger than all others

        try:
            # Top-left: infis
            infis = (ImageMobject(IMG_INFIS)
                     .scale_to_fit_height(CORNER_H)
                     .to_corner(UL, buff=CORNER_BUFF))

            # Top-right: ntc
            ntc_logo = (ImageMobject(IMG_NTC)
                        .scale_to_fit_height(CORNER_H)
                        .to_corner(UR, buff=CORNER_BUFF))

            # Bottom: vlajky + interreg side by side, same height, centred
            vlajky = ImageMobject(IMG_VLAJKY).scale_to_fit_height(INTERREG_H)
            interreg_logo = ImageMobject(IMG_INTERREG).scale_to_fit_height(INTERREG_H)
            bottom_pair = Group(vlajky, interreg_logo).arrange(RIGHT, buff=0.5)
            bottom_pair.to_edge(DOWN, buff=0.4).set_x(0)

            disclaimer = Text(
                "Seminář je podpořen z Česko-Bavorského projektu QuantumNetX Interreg BYCZ04-183 "
                "a podporuje spolupráci ve vzdělávání a síťování mezi Českem a Bavorskem "
                "v oblasti kvantových technologií.",
                font_size=FS_TINY,
                color=GRAY,
                line_spacing=1.2,
            ).set_width(config.frame_width - 1.0)
            disclaimer.next_to(bottom_pair, UP, buff=0.3)

            all_logos = Group(infis, ntc_logo, bottom_pair, disclaimer)

        except OSError:
            all_logos = Text("(Loga nenalezena)", font_size=FS_BODY, color=RED).to_edge(DOWN)

        self.play(FadeIn(text_group, shift=UP * 0.5), run_time=1.5)
        self.play(FadeIn(all_logos,  shift=UP * 0.3), run_time=1.0)
        self.next_slide(notes="Titulní snímek.")
        self.play(FadeOut(text_group), FadeOut(all_logos))

    # ── slide_02: chest X-ray + pixel-wave ──────────────────────────────────
    def slide_02(self):
        cxr = ImageMobject(IMG_CXR).scale_to_fit_height(5.5)
        self.play(FadeIn(cxr))
        self.wait(3)

        # Dimension braces
        top_brace  = Brace(cxr, direction=UP)
        left_brace = Brace(cxr, direction=LEFT)
        braces_and_text = VGroup(
            top_brace,  top_brace.get_text("224 px"),
            left_brace, left_brace.get_text("224 px"),
        )
        self.play(Write(braces_and_text), run_time=1.5)
        self.wait(1.5)

        # Pixel-wave overlay
        # Image is 224×224px — use a square grid so cells are not distorted.
        # GRID_N tiles each side; cell size is derived from the scene image size.
        GRID_N = 22
        cell_w = cxr.width  / GRID_N
        cell_h = cxr.height / GRID_N   # equal because image is square

        grid = VGroup(*[
            Square(side_length=cell_w, stroke_color=YELLOW, stroke_width=2)
            .move_to(
                cxr.get_corner(UL)
                + RIGHT * cell_w * (j + 0.5)
                + DOWN  * cell_h * (i + 0.5)
            )
            for i in range(GRID_N) for j in range(GRID_N)
        ])

        # Diagonal wave sort (top-left → bottom-right)
        grid.submobjects.sort(key=lambda sq: sq.get_center()[0] - sq.get_center()[1])

        K = 0.15
        self.play(
            LaggedStart(
                *[Succession(FadeIn(sq, run_time=K), FadeOut(sq, run_time=K * 0.5))
                  for sq in grid],
                lag_ratio=0.03,
            )
        )
        self.wait(1)
        self.next_slide(notes="Snímek CXR s pixelovou mřížkou.")
        self.play(FadeOut(cxr), FadeOut(braces_and_text))

    # ── slide_03: "Neuronová Síť" word-highlight ────────────────────────────
    def slide_03(self):
        def _make_title(blue_slice, yellow_slice=None) -> Text:
            t = Text("Neuronová Síť", font_size=FS_DISPLAY)
            t[blue_slice].set_color(BLUE_D)
            if yellow_slice:
                t[yellow_slice].set_color(YELLOW_E)
            return t

        # Step 1: plain white
        text = Text("Neuronová Síť", font_size=FS_DISPLAY)
        self.play(Write(text))

        # Step 2: "Neuronová" → blue
        text_blue = _make_title(slice(0, 9))
        q1     = Text("Co je to neuron?", font_size=FS_TITLE, color=BLUE_D)
        q1.next_to(text_blue[:9], DOWN, buff=1.5).shift(LEFT * 3)
        arrow1 = Arrow(q1.get_top(), text_blue[:9].get_bottom(), color=BLUE_D)

        self.next_slide()
        self.play(Transform(text, text_blue), Write(q1), GrowArrow(arrow1))
        self.pause()

        # Step 3: "Síť" → yellow
        text_both = _make_title(slice(0, 9), slice(9, None))
        q2     = Text("Jak jsou propojeny?", font_size=FS_TITLE, color=YELLOW_E)
        q2.next_to(text_both[10:], DOWN, buff=1.5).shift(RIGHT)
        arrow2 = Arrow(q2.get_top(), text_both[10:].get_bottom(), color=YELLOW_E)

        self.next_slide()
        self.play(Transform(text, text_both), Write(q2), GrowArrow(arrow2))
        self.next_slide()
        self.play(FadeOut(text, q1, q2, arrow1, arrow2))

    # ── slide_04: neuron / activation demo ──────────────────────────────────
    def slide_04(self):
        tracker = ValueTracker(0.42)

        neuron_text = Text("Neuron", font_size=FS_HERO, color=BLUE_D).move_to(LEFT * 3)
        circ        = Circle(radius=0.6, color=BLUE_D, fill_opacity=tracker.get_value())
        val         = DecimalNumber(tracker.get_value(), num_decimal_places=2, font_size=FS_BODY)
        neuron_group = VGroup(circ, val).next_to(neuron_text, UP, buff=0.5)

        arrow       = Arrow(neuron_text.get_right(), neuron_text.get_right() + RIGHT * 2,
                            color=WHITE, buff=0.2)
        explanation = Text("Věc co drží číslo", font_size=FS_LARGE).next_to(arrow, RIGHT, buff=0.2)

        # Intro
        self.play(FadeIn(neuron_text), Create(circ), Write(val))
        self.play(GrowArrow(arrow), Write(explanation))
        self.next_slide()

        # Attach updaters then move to centre
        val.add_updater(lambda d: d.set_value(tracker.get_value()))
        circ.add_updater(lambda c: c.set_fill(opacity=tracker.get_value()))

        self.play(
            FadeOut(neuron_text, arrow, explanation),
            neuron_group.animate.move_to(ORIGIN).scale(1.5),
        )
        self.pause()

        # Pulse: 0.42 → 0 → 1 → 0.41
        for target, rt in [(0.00, 1.5), (1.00, 2.0), (0.41, 1.5)]:
            self.play(tracker.animate.set_value(target), run_time=rt, rate_func=linear)
        self.pause()

        activation = Text(' - ,,Aktivace"', font_size=FS_LARGE)
        activation.next_to(neuron_group, RIGHT, buff=0.2)
        self.play(Write(activation))

        circ.clear_updaters()
        val.clear_updaters()
        self.next_slide()
        self.play(FadeOut(neuron_group), FadeOut(activation))

    # ── slide_05: neural-network diagram + weights ───────────────────────────
    def slide_05(self):
        LAYER_SIZES  = [10, 10, 10, 2]
        NODE_RADIUS  = 0.15
        EDGE_OPACITY = 0.4

        # ---- Build nodes ------------------------------------------------
        layers       = VGroup()
        neuron_layers: list[list] = []

        for i, size in enumerate(LAYER_SIZES):
            layer   = VGroup()
            neurons = []

            if i == 0:
                # Input layer: split with vertical ellipsis in the middle
                half = size // 2
                for _ in range(half):
                    n = Circle(radius=NODE_RADIUS, fill_opacity=0.0, color=WHITE)
                    layer.add(n); neurons.append(n)
                layer.add(Tex(r"$\vdots$").scale(1.5))
                for _ in range(half):
                    n = Circle(radius=NODE_RADIUS, fill_opacity=0.0, color=WHITE)
                    layer.add(n); neurons.append(n)
            else:
                for _ in range(size):
                    n = Circle(radius=NODE_RADIUS, fill_opacity=0.0, color=WHITE)
                    layer.add(n); neurons.append(n)

            layer.arrange(DOWN, buff=0.2)
            layers.add(layer)
            neuron_layers.append(neurons)

        layers.arrange(RIGHT, buff=2)

        # ---- Build edges (touching circle boundaries) -------------------
        edges = VGroup()
        for i in range(len(neuron_layers) - 1):
            for n1 in neuron_layers[i]:
                for n2 in neuron_layers[i + 1]:
                    c1, c2 = n1.get_center(), n2.get_center()
                    d = (c2 - c1) / np.linalg.norm(c2 - c1)
                    line = Line(
                        c1 + d * NODE_RADIUS,
                        c2 - d * NODE_RADIUS,
                        stroke_width=3.5,
                        stroke_opacity=EDGE_OPACITY,
                    ).set_z_index(-1)
                    edges.add(line)

        # ---- Show network -----------------------------------------------
        self.play(Create(layers))
        self.play(LaggedStart(*[Create(e) for e in edges], lag_ratio=0.02), run_time=2)
        self.next_slide()

        # ---- Highlight sections -----------------------------------------
        sections = {
            "Input":  layers[:1],
            "Hidden": VGroup(layers[1], layers[2]),
            "Output": layers[3:],
        }
        for _, group in sections.items():
            box = SurroundingRectangle(group, color=YELLOW, buff=0.2, stroke_width=4)
            self.play(Create(box))
            self.next_slide()
            self.play(FadeOut(box))

        # ---- Zoom into one neuron ---------------------------------------
        target_neuron = layers[1][4]

        # Edges that connect TO the target neuron (keep); all others fade out
        kept_edges = VGroup(*(
            e for e in edges
            if np.linalg.norm(e.get_end() - target_neuron.get_center()) <= 0.2
        ))
        fade_edges = VGroup(*(
            e for e in edges
            if np.linalg.norm(e.get_end() - target_neuron.get_center()) > 0.2
        ))
        fade_nodes = VGroup(*(
            node
            for i, layer in enumerate(layers) if i != 0
            for node in layer if node is not target_neuron
        ))

        self.play(FadeOut(fade_nodes), FadeOut(fade_edges), run_time=1.5)

        # ---- Pixel grid -------------------------------------------------
        ROWS, COLS = 25, 25
        GRID_UNIT  = 5.0
        pixel_side = GRID_UNIT / COLS

        pixels = VGroup(*[
            Square(side_length=pixel_side,
                   fill_color=BLACK, fill_opacity=1.0,
                   stroke_color=COLOR_PIXEL_STROKE, stroke_width=0.5)
            for _ in range(ROWS * COLS)
        ])
        pixels.arrange_in_grid(rows=ROWS, cols=COLS, buff=0)

        border     = SurroundingRectangle(pixels, color=COLOR_GRID_BORDER, buff=0.05, stroke_width=3)
        grid_group = VGroup(pixels, border).scale_to_fit_height(layers[0].height * 0.85)

        # Don't wrap kept_edges into a new VGroup — animating them directly
        # avoids re-adding faded mobjects back into the scene tree
        remaining_nn = VGroup(layers[0], target_neuron)

        nn_shift = LEFT * 2.5
        grid_group.move_to(target_neuron.get_center() + nn_shift + RIGHT * 7.0)

        self.play(
            remaining_nn.animate.shift(nn_shift),
            kept_edges.animate.shift(nn_shift),
            FadeIn(grid_group, shift=RIGHT * 0.5),
            run_time=2,
        )
        self.next_slide()

        # ---- Light up pixels & neurons ----------------------------------
        lit_pixels  = VGroup(*(pixels[i * COLS + j] for i in range(8, 12) for j in range(9, 16)))
        lit_neurons = VGroup(layers[0][0], layers[0][1], layers[0][-2], layers[0][-1])

        self.play(
            lit_pixels.animate.set_fill(WHITE, opacity=1.0).set_stroke(WHITE, width=0.5),
            lit_neurons.animate.set_fill(WHITE, opacity=1.0),
            run_time=1.5,
        )
        self.next_slide()

        # ---- Weight column ----------------------------------------------
        weight_data = [
            ("w_1", "2.07"),  ("w_2",  "2.31"),  ("w_3",  "3.64"),
            ("w_4", "1.87"),  ("w_5",  "-1.51"), ("w_6",  "-0.43"),
            ("w_7", "2.01"),  ("w_8",  "1.07"),  ("w_9",  "-0.89"),
            ("w_{10}", "1.42"),
        ]

        weights_col      = VGroup()
        row_animations   = []

        for i, (w_sym, w_val) in enumerate(weight_data):
            color = GREEN if float(w_val) >= 0 else RED
            row   = MathTex(w_sym, ":", w_val)
            row[0].set_color(color)
            row[2].set_color(color)
            weights_col.add(row)

            anim = AnimationGroup(FadeIn(row, shift=LEFT * 0.2))
            if i < len(kept_edges):
                anim = AnimationGroup(
                    FadeIn(row, shift=LEFT * 0.2),
                    kept_edges[i].animate.set_color(color).set_stroke(opacity=0.8),
                )
            row_animations.append(anim)

        dots = MathTex(r"\vdots", color=WHITE)
        weights_col.add(dots)
        row_animations.append(FadeIn(dots))

        weights_col.arrange(DOWN, aligned_edge=LEFT, buff=0.2).scale(0.7)

        weights_title = Text("Váhy", color=WHITE, font_size=FS_BASE)
        weights_title.next_to(weights_col, UP, buff=0.5)
        weight_arrow  = Arrow(
            weights_title.get_left() + LEFT * 0.2,
            weights_col.get_left()   + LEFT * 0.2 + UP * 1.5,
            color=WHITE, buff=0, stroke_width=4,
        )
        VGroup(weights_title, weights_col, weight_arrow) \
            .next_to(target_neuron, RIGHT, buff=0.4) \
            .shift(UP * 0.2)

        self.play(FadeIn(weights_title), GrowArrow(weight_arrow))
        self.play(LaggedStart(*row_animations, lag_ratio=0.15), run_time=3.5)
        self.next_slide()

        # ---- Colour pixels green / red ----------------------------------
        self.play(
            LaggedStart(
                *[p.animate.set_fill(GREEN if k % 2 == 0 else RED)
                          .set_stroke(GREEN if k % 2 == 0 else RED, width=0.5)
                  for k, p in enumerate(lit_pixels)],
                lag_ratio=0.05,
            ),
            run_time=2,
        )
        self.next_slide()
        _fade_all(self)

    # ── slide_06: weighted sum + number-line squeeze ─────────────────────────
    def slide_06(self):
        formula = MathTex(
            "a_0", r"\cdot", "w_0", "+",
            "a_1", r"\cdot", "w_1", "+",
            r"\dots", "+",
            "a_n", r"\cdot", "w_n",
            font_size=FS_HERO,
        )
        # Colour every weight term green (indices 2, 6, 12)
        for idx in (2, 6, 12):
            formula[idx].set_color(GREEN)
        formula.to_edge(UP, buff=0.5)
        self.play(Write(formula), run_time=1.5)
        self.next_slide()

        # Full-width number line
        number_line = NumberLine(x_range=[-7, 7, 1], length=13.5, color=BLUE,
                                 include_numbers=True, label_direction=DOWN)
        number_line.shift(UP * 0.5)
        for num in number_line.numbers:
            num.set_color(WHITE)

        tracker = ValueTracker(-7)
        sweep_arrow = always_redraw(lambda: Arrow(
            start=formula.get_bottom() + DOWN * 0.2,
            end=number_line.n2p(tracker.get_value()) + UP * 0.2,
            color=YELLOW, buff=0, stroke_width=5,
        ))

        self.play(Create(number_line), GrowArrow(sweep_arrow), run_time=2)
        self.play(tracker.animate.set_value(7), run_time=4, rate_func=linear)
        self.next_slide()
        self.play(FadeOut(sweep_arrow))

        # Second number line – compressed [0, 1] view
        number_line_2 = NumberLine(x_range=[-7, 7, 1], length=13.5, color=WHITE,
                                   include_numbers=True, label_direction=DOWN)
        number_line_2.next_to(number_line, DOWN, buff=2.0)
        self.play(Create(number_line_2), run_time=1.5)
        self.next_slide()

        interval   = Line(number_line_2.n2p(0), number_line_2.n2p(1),
                          color=YELLOW, stroke_width=12)
        highlight  = SurroundingRectangle(interval, color=YELLOW, buff=0.1,
                                          stroke_width=2, fill_color=YELLOW, fill_opacity=0.2)
        self.play(Create(interval), FadeIn(highlight), run_time=1.5)
        self.next_slide()

        # Squeeze animation
        funnel_l = DashedLine(number_line.n2p(-7), number_line_2.n2p(0),
                              color=GRAY, stroke_width=2)
        funnel_r = DashedLine(number_line.n2p(7),  number_line_2.n2p(1),
                              color=GRAY, stroke_width=2)
        self.play(Create(funnel_l), Create(funnel_r), run_time=1.5)
        self.next_slide()

        line_copy = number_line.copy()
        self.add(line_copy)
        self.play(
            line_copy.animate.scale(1 / 14).move_to(number_line_2.n2p(0.5)).set_color(YELLOW),
            line_copy.numbers.animate.set_opacity(0),
            run_time=3,
        )
        self.next_slide()
        _fade_all(self)

    # ── slide_07: sigmoid function + graph ───────────────────────────────────
    def slide_07(self):
        formula = MathTex(r"\sigma(x) = \frac{1}{1 + e^{-x}}", font_size=FS_HERO)
        self.play(Write(formula), run_time=1.5)
        self.next_slide()
        self.play(formula.animate.to_corner(UL, buff=1.2))

        axes = Axes(
            x_range=[-6, 6, 1],
            y_range=[-0.2, 1.2, 0.5],
            x_length=10, y_length=6,
            axis_config={"color": WHITE},
            y_axis_config={"numbers_to_include": [0, 1]},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label=r"\sigma(x)")
        self.play(Create(axes), Write(axes_labels), run_time=2)
        self.next_slide()

        # Asymptotes
        for y_val in (0, 1):
            self.play(Create(DashedLine(axes.c2p(-6, y_val), axes.c2p(6, y_val),
                                        color=GRAY, stroke_width=2)))

        sigmoid_curve = axes.plot(lambda x: 1 / (1 + np.exp(-x)),
                                  color=BLUE_D, stroke_width=5)
        self.play(Create(sigmoid_curve), run_time=2.5)
        self.next_slide()
        _fade_all(self)

    # ── slide_08: full formula with bias insertion ───────────────────────────
    def slide_08(self):
        formula = MathTex(
            r"\sigma \left( ",          # [0]
            "a_0", r"\cdot", "w_0",    # [1–3]
            " + ",                      # [4]
            "a_1", r"\cdot", "w_1",    # [5–7]
            " + ",                      # [8]
            r"\dots",                   # [9]
            " + ",                      # [10]
            "a_n", r"\cdot", "w_n",    # [11–13]
            r"\right)",                 # [14]
            font_size=FS_DISPLAY,
        )
        for idx in (3, 7, 13):
            formula[idx].set_color(GREEN)
        formula.move_to(ORIGIN)

        self.play(Write(formula), run_time=2.5)
        self.next_slide()

        # Insert bias term by pushing the closing bracket right
        bias_term = MathTex(r" + ", "b", font_size=FS_DISPLAY)
        bias_term[1].set_color(BLUE)
        bias_term.move_to(formula[14].get_center(), aligned_edge=LEFT)

        self.play(
            formula[14].animate.shift(RIGHT * 1.2),
            FadeIn(bias_term, shift=RIGHT * 0.5),
            run_time=1.5,
        )
        self.next_slide()
        _fade_all(self)

    # ── slide_09: σ(W·a + b) matrix form ────────────────────────────────────
    def slide_09(self):
        # ---- Build mobjects ---------------------------------------------
        W_matrix = Matrix(
            [["w_{11}", "w_{12}", "w_{13}"],
             ["w_{21}", "w_{22}", "w_{23}"],
             ["w_{31}", "w_{32}", "w_{33}"]],
            left_bracket="(", right_bracket=")",
        ).set_color(W_COLOR)

        a_vector = Matrix(
            [["a_1"], ["a_2"], ["a_3"]],
            left_bracket="(", right_bracket=")",
        ).set_color(A_COLOR)

        plus_sign = MathTex("+").scale(1.4)

        b_vector = Matrix(
            [["b_1"], ["b_2"], ["b_3"]],
            left_bracket="(", right_bracket=")",
        ).set_color(B_COLOR)

        sig_open  = MathTex(r"\sigma\!\left(").scale(1.6).set_color(SIG_COLOR)
        sig_close = MathTex(r"\right)").scale(1.6).set_color(SIG_COLOR)

        # ---- Layout: build full expression centred then reveal in stages
        inner = VGroup(W_matrix, a_vector, plus_sign, b_vector).arrange(RIGHT, buff=0.35)
        inner.move_to(ORIGIN)
        sig_open.next_to(inner, LEFT, buff=0.15)
        sig_close.next_to(inner, RIGHT, buff=0.15)

        label_W = _make_label("Weight matrix", W_COLOR).next_to(W_matrix, UP, buff=0.45)
        label_a = _make_label("Input vector",  A_COLOR).next_to(a_vector, UP, buff=0.45)
        label_b = _make_label("Bias vector",   B_COLOR).next_to(b_vector, UP, buff=0.45)

        # ---- Step 1: W and a -------------------------------------------
        self.play(FadeIn(W_matrix, label_W, shift=UP * 0.3), run_time=1.2)
        self.wait(0.4)
        self.play(FadeIn(a_vector, label_a, shift=UP * 0.3), run_time=1.2)
        self.wait(1.5)
        self.next_slide()

        # ---- Step 2: + b -----------------------------------------------
        self.play(FadeOut(label_W, label_a), run_time=0.5)
        self.play(FadeIn(plus_sign, b_vector, label_b, shift=RIGHT * 0.3), run_time=1.2)
        self.wait(1.5)
        self.next_slide()

        # ---- Step 3: wrap with σ( … ) ----------------------------------
        self.play(FadeOut(label_b), run_time=0.4)
        self.play(inner.animate.shift(RIGHT * 0.5), run_time=0.6)

        # Re-anchor sigma brackets after the shift
        sig_open.next_to(inner, LEFT, buff=0.15)
        sig_close.next_to(inner, RIGHT, buff=0.15)

        self.play(
            FadeIn(sig_open,  shift=LEFT  * 0.4),
            FadeIn(sig_close, shift=RIGHT * 0.4),
            run_time=1.0,
        )

        # Brief colour pulse to draw attention
        self.play(sig_open.animate.set_color(YELLOW),
                  sig_close.animate.set_color(YELLOW), run_time=0.5)
        self.play(sig_open.animate.set_color(SIG_COLOR),
                  sig_close.animate.set_color(SIG_COLOR), run_time=0.5)

        full_expr  = VGroup(sig_open, inner, sig_close)
        final_label = MathTex(r"\text{Výstup} = \sigma(W\mathbf{a} + \mathbf{b})",
                               color=LIGHT_GREY).scale(0.65)
        final_label.next_to(full_expr, DOWN, buff=0.6)

        self.play(Write(final_label), run_time=1.2)
        self.wait(2.5)
        self.next_slide()
        self.play(FadeOut(full_expr), FadeOut(final_label), run_time=1.2)
        self.wait(0.5)

    # ── slide_10: NN edges white → coloured weights → signal forward-pass ───
    def slide_10(self):
        LAYER_SIZES = [4, 5, 5, 3]
        NODE_RADIUS = 0.22
        X_SPACING   = 2.8
        Y_SPACING   = 0.9

        # ── Build nodes ────────────────────────────────────────────────
        total_w       = (len(LAYER_SIZES) - 1) * X_SPACING
        start_x       = -total_w / 2
        layers        = VGroup()
        neuron_layers = []

        for i, size in enumerate(LAYER_SIZES):
            layer   = VGroup()
            neurons = []
            for j in range(size):
                node = Circle(radius=NODE_RADIUS, color=WHITE,
                              fill_color=BLACK, fill_opacity=1.0,
                              stroke_width=2)
                node.move_to([start_x + i * X_SPACING,
                              (j - (size - 1) / 2) * Y_SPACING, 0])
                layer.add(node)
                neurons.append(node)
            layers.add(layer)
            neuron_layers.append(neurons)

        # ── Build edges + assign random weights ────────────────────────
        # edge_groups[i] = all edges between layer i and layer i+1
        # weight_groups[i] = matching weight values in the same order
        edge_groups   = []
        weight_groups = []
        all_edges     = VGroup()

        for i in range(len(neuron_layers) - 1):
            grp     = VGroup()
            weights = []
            for n1 in neuron_layers[i]:
                for n2 in neuron_layers[i + 1]:
                    w   = random.uniform(-1.0, 1.0)
                    c1, c2 = n1.get_center(), n2.get_center()
                    d      = (c2 - c1) / np.linalg.norm(c2 - c1)
                    line   = Line(
                        c1 + d * NODE_RADIUS,
                        c2 - d * NODE_RADIUS,
                        stroke_width=2.0,
                        stroke_opacity=0.5,
                        color=WHITE,
                    ).set_z_index(-1)
                    grp.add(line)
                    weights.append(w)
            edge_groups.append(grp)
            weight_groups.append(weights)
            all_edges.add(grp)

        # ── SLIDE 1: network with all-white edges ───────────────────────
        self.play(
            LaggedStart(*[FadeIn(layer) for layer in layers], lag_ratio=0.2),
            run_time=1.0,
        )
        self.play(
            LaggedStart(*[Create(e) for e in all_edges], lag_ratio=0.01),
            run_time=1.8,
        )
        self.next_slide()

        # ── SLIDE 2: edges coloured by weight value ─────────────────────
        colour_anims = []
        for grp, weights in zip(edge_groups, weight_groups):
            for edge, w in zip(grp, weights):
                colour_anims.append(
                    edge.animate.set_color(_weight_color(w)).set_stroke(opacity=0.85)
                )
        self.play(LaggedStart(*colour_anims, lag_ratio=0.008), run_time=1.5)
        self.next_slide()

        # ── SLIDE 3: forward pass with real sigmoid activations ────────
        # Each neuron holds a continuous value in [0, 1].
        # Fill opacity reflects that value directly — 0.0 = black, 1.0 = white.
        # Signal dots are white.

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        def _set_activation(node, value: float):
            """Animate a node to reflect its activation as fill opacity."""
            return node.animate(run_time=0.35).set_fill(WHITE, opacity=float(value))

        def _travel(edge):
            """White dot travelling along an edge."""
            dot = Dot(radius=0.07, color=WHITE, z_index=2)
            dot.move_to(edge.get_start())
            return Succession(
                FadeIn(dot, run_time=0.04),
                MoveAlongPath(dot, edge, run_time=0.4, rate_func=linear),
                FadeOut(dot, run_time=0.04),
            )

        # Random input activations in (0, 1) for the first layer
        current_acts = [random.uniform(0.1, 1.0) for _ in neuron_layers[0]]

        # Show input layer activations immediately
        self.play(
            AnimationGroup(*[
                _set_activation(neuron_layers[0][j], v)
                for j, v in enumerate(current_acts)
            ]),
            run_time=0.5,
        )

        for layer_idx in range(len(edge_groups)):
            next_size  = len(neuron_layers[layer_idx + 1])
            src_size   = len(neuron_layers[layer_idx])
            weights    = weight_groups[layer_idx]   # flat list, row-major n1→n2

            # ── Travel: send dots along ALL edges, scaled opacity by
            #    source activation so dimmer neurons send dimmer signals
            travel_anims = []
            for j, src_act in enumerate(current_acts):
                if src_act > 0.05:   # skip near-zero neurons
                    for k in range(next_size):
                        travel_anims.append(
                            _travel(edge_groups[layer_idx][j * next_size + k])
                        )

            if travel_anims:
                self.play(LaggedStart(*travel_anims, lag_ratio=0.02), run_time=1.1)

            # ── Compute next layer via weighted sum + sigmoid ────────────
            next_acts = []
            for k in range(next_size):
                z = sum(
                    current_acts[j] * weights[j * next_size + k]
                    for j in range(src_size)
                )
                next_acts.append(_sigmoid(z))

            # ── Dim current layer down, light next layer up ──────────────
            dim_anims = [
                _set_activation(neuron_layers[layer_idx][j], 0.0)
                for j in range(src_size)
            ]
            act_anims = [
                _set_activation(neuron_layers[layer_idx + 1][k], next_acts[k])
                for k in range(next_size)
            ]
            self.play(AnimationGroup(*dim_anims, *act_anims), run_time=0.45)

            current_acts = next_acts

        # Keep final activations as a local; passed by name into the slides below.
        final_acts = current_acts

        self.wait(0.8)
        self.next_slide()

        # ── SLIDE 4: zoom entire network so output layer lands on left ─
        output_layer = layers[-1]

        # Whole-network group keeps every object moving together
        entire_network = Group(layers, all_edges)

        rest = Group(
            *[layers[i] for i in range(len(layers) - 1)],
            all_edges,
        )

        # How far from the left edge the output column should land (scene units).
        # Chosen so the output nodes sit in the left quarter of the frame with
        # comfortable padding before the screen edge.
        ZOOM_TARGET_X_OFFSET = 2.5
        # Magnification factor — tuned so 3 output nodes fill ~40% of frame height.
        ZOOM_SCALE           = 2.8

        target_x     = -config.frame_width / 2 + ZOOM_TARGET_X_OFFSET
        scale_factor = ZOOM_SCALE

        # Where the output layer centre is RIGHT NOW (before any transform)
        output_center = output_layer.get_center()

        # After scaling the whole network by scale_factor the output layer
        # centre moves to:  network_centre + (output_center - network_centre) * scale_factor
        # We want that final position to be at target_x, so we compute the
        # required translation of the whole network.
        network_center = entire_network.get_center()
        scaled_output_x = network_center[0] + (output_center[0] - network_center[0]) * scale_factor
        dx = target_x - scaled_output_x

        # Step 1 — zoom the whole network; output layer ends up on the left
        self.play(
            entire_network.animate
                .scale(scale_factor)
                .shift([dx, 0, 0]),
            run_time=1.4,
            rate_func=smooth,
        )

        self.wait(0.3)

        # Step 2 — fade rest, leaving only the output layer
        self.play(FadeOut(rest), run_time=0.9)

        self.wait(0.5)
        self.next_slide()

        # ── SLIDE 5: duplicate output neurons as targets, show loss ─────
        predicted_nodes = neuron_layers[-1]
        n_out           = len(predicted_nodes)
        # Gap between predicted and target columns expressed as a multiple of the
        # output layer's width, so it scales naturally if NODE_RADIUS is changed.
        COL_GAP_FACTOR  = 1.8
        col_gap         = output_layer.width * COL_GAP_FACTOR

        target_nodes = []
        for idx, pred_node in enumerate(predicted_nodes):
            tgt = pred_node.copy()
            tgt.set_fill(WHITE, opacity=1.0 if idx == 0 else 0.0)
            tgt.move_to(pred_node.get_center() + RIGHT * col_gap)
            target_nodes.append(tgt)

        target_group = VGroup(*target_nodes)

        # Single double-headed arrow at the middle pair only
        mid      = n_out // 2
        mid_pred = predicted_nodes[mid]
        mid_tgt  = target_nodes[mid]
        mid_arrow = DoubleArrow(
            start=mid_pred.get_right() + RIGHT * 0.08,
            end=mid_tgt.get_left()     - RIGHT * 0.08,
            color=RED,
            stroke_width=3,
            tip_length=0.18,
            buff=0,
        )

        self.play(
            LaggedStart(
                *[FadeIn(n, shift=RIGHT * 0.3) for n in target_nodes],
                lag_ratio=0.15,
            ),
            GrowArrow(mid_arrow),
            run_time=1.2,
        )

        self.wait(0.8)
        self.next_slide()

        # ── SLIDE 6: MSE terms with real neuron values ───────────────────
        # Each row: (ŷ_i - y_i)² with actual sigmoid output and one-hot target
        mse_terms = VGroup()
        for idx, pred_node in enumerate(predicted_nodes):
            y_hat = final_acts[idx]      # real sigmoid activation
            y     = 1.0 if idx == 0 else 0.0

            term = MathTex(
                rf"({y_hat:.2f} - {y:.1f})^2",
                font_size=FS_LARGE,
            )
            term.next_to(target_nodes[idx], RIGHT, buff=1.4)
            term.set_y(pred_node.get_center()[1])
            mse_terms.add(term)

        self.play(
            LaggedStart(
                *[FadeIn(t, shift=DOWN * 0.2) for t in mse_terms],
                lag_ratio=0.4,
            ),
            run_time=1.2,
        )

        self.wait(0.5)
        self.next_slide()

        # ── SLIDE 7: curly brace + sum label ────────────────────────────
        # Draw a right-opening curly brace spanning all MSE terms, then
        # show the computed sum value to its right.

        mse_sum = sum(
            (final_acts[i] - (1.0 if i == 0 else 0.0)) ** 2
            for i in range(n_out)
        )

        brace = Brace(mse_terms, direction=RIGHT, color=WHITE)
        sum_label = MathTex(
            rf"= {mse_sum:.4f}",
            font_size=FS_LARGE,
        )
        sum_label.next_to(brace, RIGHT, buff=0.3)

        self.play(GrowFromCenter(brace), run_time=0.8)
        self.play(FadeIn(sum_label, shift=RIGHT * 0.3), run_time=0.7)

        self.wait(0.4)
        self.next_slide()

        # ── SLIDE 8: rapid flicker through training samples → show mean ──
        # Each sample has:
        #   - random predicted activations (predicted column)
        #   - a random one-hot target vector (target column)
        # Both columns update every frame via always_redraw.
        # At the end brace + MSE rows fade, mean centres on screen.

        N_SAMPLES = 24
        fake_losses   = []
        fake_rows     = []   # predicted (yh, y_target) per row
        fake_pred_ops = []   # predicted node opacities
        fake_tgt_hot  = []   # which target neuron is "on" (index)

        for _ in range(N_SAMPLES):
            acts    = [random.uniform(0.05, 0.95) for _ in range(n_out)]
            hot_idx = random.randint(0, n_out - 1)            # one-hot index
            targets = [1.0 if i == hot_idx else 0.0 for i in range(n_out)]
            loss    = sum((acts[i] - targets[i]) ** 2 for i in range(n_out))
            fake_losses.append(loss)
            fake_rows.append(list(zip(acts, targets)))
            fake_pred_ops.append(acts)
            fake_tgt_hot.append(hot_idx)

        mean_loss = sum(fake_losses) / len(fake_losses)

        # ── live updaters driven by a single ValueTracker ────────────────
        tracker = ValueTracker(0.0)

        # Predicted node opacities
        for node_idx, pred_node in enumerate(predicted_nodes):
            def _pred_op(m, ni=node_idx):
                s = int(tracker.get_value()) % N_SAMPLES
                m.set_fill(WHITE, opacity=fake_pred_ops[s][ni])
            pred_node.add_updater(_pred_op)

        # Target node opacities (one-hot)
        for node_idx, tgt_node in enumerate(target_nodes):
            def _tgt_op(m, ni=node_idx):
                s = int(tracker.get_value()) % N_SAMPLES
                m.set_fill(WHITE, opacity=1.0 if fake_tgt_hot[s] == ni else 0.0)
            tgt_node.add_updater(_tgt_op)

        # MSE row labels
        def _make_row_label(idx):
            def _upd():
                s      = int(tracker.get_value()) % N_SAMPLES
                yh, y  = fake_rows[s][idx]
                t = MathTex(rf"({yh:.2f} - {y:.1f})^2", font_size=FS_LARGE)
                t.next_to(target_nodes[idx], RIGHT, buff=1.4)
                t.set_y(predicted_nodes[idx].get_center()[1])
                return t
            return always_redraw(_upd)

        # Sum label
        def _make_sum_label():
            def _upd():
                s = int(tracker.get_value()) % N_SAMPLES
                t = MathTex(rf"= {fake_losses[s]:.4f}", font_size=FS_LARGE)
                t.next_to(brace, RIGHT, buff=0.3)
                return t
            return always_redraw(_upd)

        self.remove(*mse_terms, sum_label)
        live_rows = [_make_row_label(i) for i in range(n_out)]
        live_sum  = _make_sum_label()
        for lr in live_rows:
            self.add(lr)
        self.add(live_sum)

        # Single play drives everything
        self.play(
            tracker.animate.set_value(N_SAMPLES - 1),
            run_time=2.5,
            rate_func=linear,
        )
        self.wait(0.3)

        # Clear updaters so nodes stay at their last flicker state
        for node in predicted_nodes:
            node.clear_updaters()
        for node in target_nodes:
            node.clear_updaters()

        # Row labels are already removed (live); fade them out was implicit.
        # Now: fade the static MSE row MathTex objects, keep brace + nodes.
        self.remove(*live_rows, live_sum)

        # mean label positioned to the right of the brace's post-shift location
        mean_label = MathTex(
            rf"\overline{{L}} = {mean_loss:.4f}",
            font_size=FS_HERO,
            color=YELLOW,
        )
        # Place it where it will end up: next to brace shifted LEFT 1.5
        mean_label.next_to(brace, RIGHT, buff=0.35)
        mean_label.shift(LEFT * 1.5)

        self.play(
            brace.animate.shift(LEFT * 1.5),
            FadeIn(mean_label, shift=RIGHT * 0.2),
            run_time=0.9,
        )

        self.wait(1.0)
        self.next_slide()
        _fade_all(self)

    # ── slide_11: loss landscape with local and global minima ───────────────
    def slide_11(self):
        # f(x) = 0.15x^4 - 0.5x^3 - 0.6x^2 + 2x + 2
        # local minimum  near x ≈ -1.25
        # global minimum near x ≈  2.78
        def f(x):
            return 0.15*x**4 - 0.5*x**3 - 0.6*x**2 + 2*x + 2

        def df(x):
            return 0.6*x**3 - 1.5*x**2 - 1.2*x + 2

        X_MIN, X_MAX = -2.3, 3.8

        ax = Axes(
            x_range=[-2.5, 4.0, 1],
            y_range=[-1.0, 7.0, 1],
            x_length=9,
            y_length=5.5,
            axis_config={"color": WHITE, "stroke_width": 2},
            tips=False,
        ).move_to(ORIGIN)

        curve = ax.plot(f, x_range=[X_MIN, X_MAX], color=BLUE, stroke_width=3.5)

        self.play(Create(ax), run_time=1.0)
        self.play(Create(curve), run_time=1.8)
        self.next_slide()

        # ── 5 balls placed evenly across the x domain ───────────────────
        # Spread start positions so they are visually well-separated and
        # sit on clearly different slopes.
        start_xs = [-2.1, -0.5, 0.8, 1.9, 3.2]
        BALL_R   = 0.13
        BALL_COL = WHITE

        # Offset ball centre along the curve normal in scene coordinates.
        # Use finite differences on ax.c2p so the scale is always correct.
        EPS = 0.01
        def ball_pos(x):
            base  = np.array(ax.c2p(x, f(x)))
            ahead = np.array(ax.c2p(x + EPS, f(x + EPS)))
            tang  = ahead - base                   # tangent vector in scene coords
            # Rotate 90° counter-clockwise to get the upward normal
            norm  = np.array([-tang[1], tang[0], 0.0])
            norm  = norm / np.linalg.norm(norm)    # unit normal
            return base + norm * BALL_R

        balls = VGroup(*[
            Circle(radius=BALL_R, color=BALL_COL,
                   fill_color=BALL_COL, fill_opacity=0.9,
                   stroke_width=1.5)
            .move_to(ball_pos(x))
            for x in start_xs
        ])

        self.play(
            LaggedStart(*[FadeIn(b, scale=0.5) for b in balls], lag_ratio=0.15),
            run_time=1.0,
        )
        self.next_slide()

        # ── Gradient-descent roll-down simulation ───────────────────────
        LR      = 0.04
        N_STEPS = 120

        trajectories = []
        for x0 in start_xs:
            xs = [x0]
            for _ in range(N_STEPS):
                x_new = xs[-1] - LR * df(xs[-1])
                x_new = max(X_MIN, min(X_MAX, x_new))
                xs.append(x_new)
            trajectories.append(xs)

        tracker = ValueTracker(0.0)

        live_balls = []
        for traj in trajectories:
            def _make_ball(t_ref=traj):
                def _upd():
                    step = min(int(tracker.get_value()), N_STEPS)
                    ball = Circle(radius=BALL_R, color=BALL_COL,
                                  fill_color=BALL_COL, fill_opacity=0.9,
                                  stroke_width=1.5)
                    ball.move_to(ball_pos(t_ref[step]))
                    return ball
                return always_redraw(_upd)
            live_balls.append(_make_ball())

        # Remove static balls, add live ones
        self.remove(*balls)
        for lb in live_balls:
            self.add(lb)

        self.play(
            tracker.animate.set_value(N_STEPS),
            run_time=4.0,
            rate_func=linear,
        )

        self.wait(0.5)
        self.next_slide()
        _fade_all(self)

    # ── slide_12: 3D Surface loss landscape + gradient descent ──────────────
    def slide_12(self):
        # C(x, y) = sin(x)*cos(y)*exp(-0.15*(x²+y²)) + 0.08*(x²+y²)
        def C(x, y):
            return (np.sin(x) * np.cos(y) * np.exp(-0.15*(x**2+y**2))
                    + 0.08*(x**2 + y**2))

        def dCdx(x, y):
            e = np.exp(-0.15*(x**2+y**2))
            return (np.cos(x)*np.cos(y)*e
                    - 0.3*x*np.sin(x)*np.cos(y)*e + 0.16*x)

        def dCdy(x, y):
            e = np.exp(-0.15*(x**2+y**2))
            return (-np.sin(x)*np.sin(y)*e
                    - 0.3*y*np.sin(x)*np.cos(y)*e + 0.16*y)

        XY_RANGE = 4.0

        # ── 1. Set the 3D Camera Perspective ─────────────────────────────
        self.move_camera(
            phi=65 * DEGREES, 
            theta=-35 * DEGREES,  # Slightly adjusted angle for a better isometric look
            frame_center=RIGHT * 1.5 + UP * 0.2,
            run_time=1.5
        )

        # ── 2. Create 3D Axes, Labels, and Surface ───────────────────────
        axes = ThreeDAxes(
            x_range=[-XY_RANGE, XY_RANGE, 1],
            y_range=[-XY_RANGE, XY_RANGE, 1],
            z_range=[-0.5, 2.5, 1],
            x_length=7, y_length=7, z_length=3,
            axis_config={"color": WHITE, "stroke_width": 1.5}
        )

        # Add axis labels directly into the 3D space
        x_label = axes.get_x_axis_label("x").scale(0.8)
        y_label = axes.get_y_axis_label("y").scale(0.8)
        z_label = axes.get_z_axis_label(r"C(x,y)").scale(0.8)
        axis_labels = VGroup(x_label, y_label, z_label)

        # Lowered resolution to emphasize the grid lines (the "wireframe/rings")
        surface = Surface(
            lambda u, v: axes.c2p(u, v, C(u, v)),
            u_range=[-XY_RANGE, XY_RANGE],
            v_range=[-XY_RANGE, XY_RANGE],
            resolution=(32, 32)
        )
        
        # Color the faces based on height
        surface.set_fill_by_value(axes=axes, colors=[PURPLE_E, PURPLE_C, PINK], axis=2)
        
        # Make the faces highly transparent but the stroke (wireframe) bright and thick
        surface.set_style(fill_opacity=0.15, stroke_width=1.5, stroke_color=PINK)

        # ── 3. Math Formulas (Fixed to Screen) ───────────────────────────
        grad_title = MathTex(r"\nabla C(x,y) =", font_size=FS_TITLE, color=WHITE)
        grad_vec   = MathTex(
            r"\begin{pmatrix} \dfrac{\partial C}{\partial x} \\[1ex]"
            r"\dfrac{\partial C}{\partial y} \end{pmatrix}",
            font_size=FS_TITLE, color=WHITE,
        )
        delta_label = MathTex(
            r"\Delta C = -\eta \, \nabla C",
            font_size=FS_TITLE, color=WHITE,
        )
        
        rhs = VGroup(grad_title, grad_vec, delta_label)
        rhs.arrange(DOWN, buff=0.5, aligned_edge=RIGHT)
        rhs.to_corner(UR, buff=0.8)

        self.add_fixed_in_frame_mobjects(rhs)

        # ── Reveal ───────────────────────────────────────────────────────
        self.play(Create(axes), Write(axis_labels), FadeIn(surface), FadeIn(rhs), run_time=2)
        self.next_slide()

        # ── 4. Gradient descent dot on 3D map (with Glow Effect!) ────────
        # LR=0.18 converges visibly within 80 steps without overshooting the
        # minimum — smaller values look too slow, larger values oscillate.
        LR      = 0.18
        N_STEPS = 80
        # Start near the far corner of the surface so the full descent path
        # is visible and the dot travels a satisfying diagonal trajectory.
        x0, y0  = 3.2, 3.0

        traj = [(x0, y0, C(x0, y0))]
        for _ in range(N_STEPS):
            xp, yp, _ = traj[-1]
            xn = float(np.clip(xp - LR * dCdx(xp, yp), -XY_RANGE, XY_RANGE))
            yn = float(np.clip(yp - LR * dCdy(xp, yp), -XY_RANGE, XY_RANGE))
            traj.append((xn, yn, C(xn, yn)))

        tracker = ValueTracker(0.0)

        # Draw the glowing dot
        def _dot():
            step   = min(int(tracker.get_value()), N_STEPS)
            xp, yp, zp = traj[step]
            pt = axes.c2p(xp, yp, zp)
            core = Dot3D(point=pt, radius=0.08, color=YELLOW)
            glow = Dot3D(point=pt, radius=0.16, color=YELLOW, fill_opacity=0.3)
            return VGroup(glow, core)

        # Draw the glowing trail
        def _trail():
            step  = min(int(tracker.get_value()), N_STEPS)
            pts   = [axes.c2p(tx, ty, tz) for tx, ty, tz in traj[:step+1]]
            if len(pts) < 2:
                return VMobject()
            
            # Layer a thick transparent line under a thin solid line to create a glow
            glow = VMobject().set_points_as_corners(pts).set_stroke(YELLOW, width=12, opacity=0.25)
            core = VMobject().set_points_as_corners(pts).set_stroke(YELLOW, width=3, opacity=1.0)
            return VGroup(glow, core)

        live_dot   = always_redraw(_dot)
        live_trail = always_redraw(_trail)
        self.add(live_trail, live_dot)

        self.play(
            tracker.animate.set_value(N_STEPS),
            run_time=5.0,
            rate_func=linear,
        )
        self.wait(0.5)
        self.next_slide()

        # ── 5. Optional: Rotate the landscape! ───────────────────────────
        self.move_camera(theta=-125 * DEGREES, run_time=3.5)
        self.next_slide()

        # ── Clean up and reset camera for the next slide ─────────────────
        self.move_camera(
            phi=0, 
            theta=-90 * DEGREES, 
            frame_center=ORIGIN, 
            run_time=1.0
        )
        _fade_all(self)
 
    # ── slide_13: weight vector W and negative gradient vector ──────────────
    def slide_13(self):
        N = 8  # number of rows in each column

        random.seed(42)
        weights   = [random.uniform(-5,  5) for _ in range(N)]
        gradients = [random.uniform(-2,  2) for _ in range(N)]

        # Reuse the module-level _weight_color; normalise each domain to [-1, 1].
        def w_col(v):  return _weight_color(v / 5.0)   # weights ∈ [-5, 5]
        def g_col(v):  return _weight_color(v / 2.0)   # neg-grads ∈ [-2, 2]

        # ── Layout constants ──────────────────────────────────────────────
        ROW_H      = 0.72    # vertical spacing between rows
        BRACE_BUFF = 0.15    # gap between numbers and brackets
        TIP_LEN    = 0.15    # length of the square bracket tips
        NUM_FS     = 36      # font size for numbers

        top_y = (N - 1) / 2 * ROW_H

        # ── Helper: build one column with square brackets ─────────────────
        def build_column(values, color_fn):
            """Returns a VGroup of (labels, b_left, b_right)."""
            labels = VGroup()
            for i, v in enumerate(values):
                y   = top_y - i * ROW_H
                col = color_fn(v)
                lbl = MathTex(rf"{v:+.2f}", font_size=NUM_FS, color=col)
                lbl.move_to([0, y, 0])
                labels.add(lbl)

            # Draw square brackets precisely around the labels
            left = labels.get_left()[0] - BRACE_BUFF
            right = labels.get_right()[0] + BRACE_BUFF
            top = labels.get_top()[1] + BRACE_BUFF
            bottom = labels.get_bottom()[1] - BRACE_BUFF

            b_left = VMobject(color=WHITE).set_points_as_corners([
                [left + TIP_LEN, top, 0],
                [left, top, 0],
                [left, bottom, 0],
                [left + TIP_LEN, bottom, 0]
            ])
            b_right = VMobject(color=WHITE).set_points_as_corners([
                [right - TIP_LEN, top, 0],
                [right, top, 0],
                [right, bottom, 0],
                [right - TIP_LEN, bottom, 0]
            ])
            
            return labels, b_left, b_right

        # Generate elements
        w_labels, w_bl, w_br = build_column(weights,                    w_col)
        g_labels, g_bl, g_br = build_column([-g for g in gradients],    g_col)

        w_vec = VGroup(w_labels, w_bl, w_br)
        g_vec = VGroup(g_labels, g_bl, g_br)

        # ── Titles placed to the LEFT of the vectors ──────────────────────
        w_title = MathTex(r"\mathbf{W} =", font_size=FS_LARGE, color=WHITE)
        g_title = MathTex(r"-\nabla C =", font_size=FS_LARGE, color=WHITE)

        w_group = VGroup(w_title, w_vec).arrange(RIGHT, buff=0.3)
        g_group = VGroup(g_title, g_vec).arrange(RIGHT, buff=0.3)

        # Center both columns side by side
        both_groups = VGroup(w_group, g_group).arrange(RIGHT, buff=1.8)
        both_groups.move_to(DOWN * 0.2)

        # ── Update rule label ─────────────────────────────────────────────
        update_rule = MathTex(
            r"\mathbf{W} \leftarrow \mathbf{W} - \eta\,\nabla C",
            font_size=FS_SLIDE, color=YELLOW,
        )
        update_rule.to_edge(UP, buff=0.35)

        # ── Animate ───────────────────────────────────────────────────────
        # Step 1: W vector
        self.play(
            LaggedStart(*[Write(l) for l in w_labels], lag_ratio=0.08),
            FadeIn(w_bl, w_br),
            FadeIn(w_title, shift=RIGHT * 0.2),
            run_time=1.2,
        )
        self.next_slide()

        # Step 2: -∇C vector
        self.play(
            LaggedStart(*[Write(l) for l in g_labels], lag_ratio=0.08),
            FadeIn(g_bl, g_br),
            FadeIn(g_title, shift=RIGHT * 0.2),
            run_time=1.2,
        )
        self.next_slide()

        # Step 3: update rule
        self.play(Write(update_rule), run_time=0.8)

        self.wait(0.8)
        self.next_slide()
        _fade_all(self)

    # ── slide_14: True Mathematical Backpropagation & Looping Wave ────────────
    def slide_14(self):
        LAYER_SIZES = [4, 5, 5, 3]
        NODE_RADIUS = 0.22
        X_SPACING   = 2.8
        Y_SPACING   = 0.9

        # ── 1. REAL NEURAL NETWORK MATH (NumPy) ───────────────────────
        np.random.seed(42)
        random.seed(42)

        # Initialize exact weight matrices W^{(l)}
        weights = []
        for i in range(len(LAYER_SIZES) - 1):
            # Shape: (next_layer_size, current_layer_size)
            w = np.random.uniform(-1.0, 1.0, (LAYER_SIZES[i+1], LAYER_SIZES[i]))
            weights.append(w)

        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        def d_sigmoid(a): return a * (1.0 - a)

        # Forward pass
        a0 = np.random.uniform(0.1, 0.9, LAYER_SIZES[0])
        acts = [a0]
        for w in weights:
            acts.append(sigmoid(w @ acts[-1]))

        # Target one-hot vector (Let's say the first class is the true class)
        target_vec = np.array([1.0, 0.0, 0.0])

        # Backward pass (Compute all actual errors/deltas for every node)
        node_errors = [None] * len(LAYER_SIZES)
        node_errors[3] = acts[3] - target_vec
        
        for l in range(2, -1, -1):
            # δ^{(l)} = (W^{(l)})^T δ^{(l+1)} ⊙ σ'(a^{(l)})
            node_errors[l] = (weights[l].T @ node_errors[l+1]) * d_sigmoid(acts[l])

        # ── Build nodes ────────────────────────────────────────────────
        total_w       = (len(LAYER_SIZES) - 1) * X_SPACING
        start_x       = -total_w / 2
        layers        = VGroup()
        neuron_layers = []

        for i, size in enumerate(LAYER_SIZES):
            layer   = VGroup()
            neurons = []
            for j in range(size):
                node = Circle(radius=NODE_RADIUS, color=WHITE,
                              fill_color=BLACK, fill_opacity=1.0,
                              stroke_width=2)
                node.move_to([start_x + i * X_SPACING,
                              (j - (size - 1) / 2) * Y_SPACING, 0])
                layer.add(node)
                neurons.append(node)
            layers.add(layer)
            neuron_layers.append(neurons)

        # ── Build edges (Using true mathematical weights) ──────────────
        edge_groups   = []
        all_edges     = VGroup()

        for i in range(len(neuron_layers) - 1):
            grp = VGroup()
            for j, n1 in enumerate(neuron_layers[i]):
                for k, n2 in enumerate(neuron_layers[i + 1]):
                    w   = weights[i][k, j]
                    c1, c2 = n1.get_center(), n2.get_center()
                    d      = (c2 - c1) / np.linalg.norm(c2 - c1)
                    line   = Line(
                        c1 + d * NODE_RADIUS,
                        c2 - d * NODE_RADIUS,
                        stroke_width=2.0,
                        stroke_opacity=0.85,
                        color=_weight_color(w),
                    ).set_z_index(-1)
                    grp.add(line)
            edge_groups.append(grp)
            all_edges.add(grp)

        # ── Build Answer Vector (Target Nodes) ──────────────────────────
        predicted_nodes = neuron_layers[-1]
        n_out           = len(predicted_nodes)
        # Scene-unit gap between the output column and the target column.
        # 2.0 gives enough room for the double-headed error arrows to breathe.
        col_gap         = 2.0

        target_nodes = []
        for idx, pred_node in enumerate(predicted_nodes):
            tgt = pred_node.copy()
            tgt.move_to(pred_node.get_center() + RIGHT * col_gap)
            target_nodes.append(tgt)

        target_group = VGroup(*target_nodes)

        error_arrows = VGroup(*[
            DoubleArrow(
                start=predicted_nodes[i].get_right() + RIGHT * 0.08,
                end=target_nodes[i].get_left() - RIGHT * 0.08,
                color=BLUE if node_errors[3][i] >= 0 else RED,
                stroke_width=2, tip_length=0.12, buff=0
            ) for i in range(n_out)
        ])

        # ── Combine into World & Pre-calculate Zoom ─────────────────────
        world = VGroup(layers, all_edges, target_group, error_arrows)

        for layer in layers:
            for node in layer: node.set_stroke(opacity=0).set_fill(opacity=0)
        for grp in all_edges:
            for edge in grp: edge.set_stroke(opacity=0)
        for node in target_group: node.set_stroke(opacity=0).set_fill(opacity=0)
        for arrow in error_arrows: arrow.set_stroke(opacity=0).set_fill(opacity=0)

        current_s = 1.0
        current_c = ORIGIN

        x_3 = start_x + 3 * X_SPACING
        x_tgt = x_3 + col_gap
        # Add 4.0 scene units of padding around the visible region so the
        # outermost nodes and arrows never sit flush against the frame edge.
        w_3 = (x_tgt - x_3) + 4.0
        
        target_c = np.array([(x_tgt + x_3) / 2, 0, 0])
        target_s = min(2.5, config.frame_width * 0.85 / w_3)

        world.scale(target_s, about_point=ORIGIN)
        world.shift(-target_c * target_s)
        
        current_s = target_s
        current_c = target_c

        self.add(world)

        # ── SLIDE 1: Reveal Output & Targets (Real Predictions) ─────────
        self.play(
            AnimationGroup(*[
                node.animate.set_stroke(opacity=1.0).set_fill(WHITE, opacity=acts[3][i])
                for i, node in enumerate(layers[3])
            ]),
            AnimationGroup(*[
                node.animate.set_stroke(opacity=1.0).set_fill(WHITE, opacity=target_vec[i])
                for i, node in enumerate(target_group)
            ]),
            run_time=1.0
        )
        self.next_slide()

        # ── SLIDE 2: Compute Actual Error ───────────────────────────────
        self.play(
            AnimationGroup(*[
                arrow.animate.set_stroke(opacity=1.0).set_fill(opacity=1.0)
                for arrow in error_arrows
            ]),
            run_time=0.8
        )
        
        def _set_error(node, val: float, mult=4.0):
            color = BLUE if val >= 0 else RED
            op = min(1.0, abs(val) * mult)
            return node.animate(run_time=0.35).set_fill(color, opacity=op)

        self.play(
            AnimationGroup(*[
                _set_error(layers[3][i], node_errors[3][i])
                for i in range(n_out)
            ]),
            run_time=1.0
        )
        self.next_slide()

        # ── HELPER FUNCTION FOR BACKPROP SIGNALS ────────────────────────
        def _travel_backwards(edge, val: float, speed_mult=1.0):
            color = BLUE if val >= 0 else RED
            dot = Dot(radius=0.07, color=color, z_index=2)
            rev_edge = edge.copy().reverse_direction()
            dot.move_to(rev_edge.get_start())
            return Succession(
                FadeIn(dot, run_time=0.04 * speed_mult),
                MoveAlongPath(dot, rev_edge, run_time=0.4 * speed_mult, rate_func=linear),
                FadeOut(dot, run_time=0.04 * speed_mult),
            )

        # ── SLIDE 3+: Step-by-Step Mathematical Backprop ────────────────
        for layer_idx in range(2, -1, -1): 
            do_zoom = False
            reveal_start = layer_idx
            
            # 2 Camera Zooms only!
            if layer_idx == 2:
                do_zoom = True
                reveal_start = 2
            elif layer_idx == 1:
                do_zoom = True
                reveal_start = 0 
                
            if do_zoom:
                x_k = start_x + reveal_start * X_SPACING
                w_k = (x_tgt - x_k) + 4.0
                target_c = np.array([(x_tgt + x_k) / 2, 0, 0])
                target_s = min(2.5, config.frame_width * 0.85 / w_k)
                
                world.generate_target()
                world.target.shift(current_c * current_s)
                world.target.scale(1 / current_s, about_point=ORIGIN)
                world.target.scale(target_s, about_point=ORIGIN)
                world.target.shift(-target_c * target_s)
                
                for r_idx in range(reveal_start, layer_idx + 1):
                    for node in world.target[0][r_idx]:
                        node.set_stroke(opacity=1.0)
                    for edge in world.target[1][r_idx]:
                        edge.set_stroke(opacity=0.85)
                    
                self.play(MoveToTarget(world), run_time=1.5, rate_func=smooth)
                current_s, current_c = target_s, target_c
                self.next_slide()
            
            # Send true mathematical signals across edges
            dest_size = len(neuron_layers[layer_idx])     # Left layer
            src_size  = len(neuron_layers[layer_idx + 1]) # Right layer

            travel_anims = []
            for j in range(dest_size):
                for k in range(src_size):
                    edge = edge_groups[layer_idx][j * src_size + k]
                    signal_val = weights[layer_idx][k, j] * node_errors[layer_idx+1][k]
                    travel_anims.append(_travel_backwards(edge, signal_val))

            if travel_anims:
                self.play(LaggedStart(*travel_anims, lag_ratio=0.02), run_time=1.1)

            # Dim right layer, light up left layer using computed deltas
            dim_anims = [
                _set_error(neuron_layers[layer_idx + 1][j], 0.0)
                for j in range(src_size)
            ]
            act_anims = [
                _set_error(neuron_layers[layer_idx][k], node_errors[layer_idx][k])
                for k in range(dest_size)
            ]
            self.play(AnimationGroup(*dim_anims, *act_anims), run_time=0.45)
            
            if layer_idx > 0:
                self.next_slide()

        # Final input layer cleanup 
        self.play(
            AnimationGroup(*[_set_error(node, 0.0) for node in neuron_layers[0]]),
            run_time=0.5
        )


        # ── SECOND-TO-LAST SLIDE: INFINITE LOOPING BACKPROP WAVE ────────
        self.next_slide(loop=True)

        # 1. Reset: Light up Output, Dim Input
        out_anims = [_set_error(neuron_layers[3][i], node_errors[3][i]) for i in range(LAYER_SIZES[3])]
        in_anims  = [_set_error(neuron_layers[0][i], 0.0) for i in range(LAYER_SIZES[0])]
        self.play(AnimationGroup(*(out_anims + in_anims)), run_time=0.4)

        # 2. Continuous rapid wave right to left
        for layer_idx in range(2, -1, -1):
            dest_size = len(neuron_layers[layer_idx])
            src_size  = len(neuron_layers[layer_idx+1])
            travel_anims = []
            
            for j in range(dest_size):
                for k in range(src_size):
                    edge = edge_groups[layer_idx][j * src_size + k]
                    signal_val = weights[layer_idx][k, j] * node_errors[layer_idx+1][k]
                    travel_anims.append(_travel_backwards(edge, signal_val, speed_mult=0.6))

            self.play(LaggedStart(*travel_anims, lag_ratio=0.01), run_time=0.6)

            dim_anims = [_set_error(neuron_layers[layer_idx+1][k], 0.0) for k in range(src_size)]
            act_anims = [_set_error(neuron_layers[layer_idx][j], node_errors[layer_idx][j]) for j in range(dest_size)]
            self.play(AnimationGroup(*(dim_anims + act_anims)), run_time=0.25)

        self.wait(0.3)


        # ── GRACEFUL EXIT ───────────────────────────────────────────────
        self.next_slide()
        _fade_all(self)

    # ── slide_15: Architecture Pipeline Scheme ──────────────────────────────
    def slide_15(self):

        title = Text("Architektura Modelu", font_size=FS_LARGE, weight=BOLD).to_edge(UP, buff=0.4)
        
        # Helper to create styled nodes
        def make_node(text, color, width=2.4, height=1.0):
            box = RoundedRectangle(corner_radius=0.2, width=width, height=height, 
                                   color=color, fill_opacity=0.2, stroke_width=3)
            label = Text(text, font_size=FS_BODY).move_to(box)
            return VGroup(box, label)

        # ── 1. CXR Input ────────────────────────────────────────────────
        try:
            cxr_img = ImageMobject(IMG_CXR).scale_to_fit_height(1.8)
        except OSError:
            # Fallback if image isn't found during rendering
            cxr_img = RoundedRectangle(corner_radius=0.1, width=1.8, height=1.8, color=GRAY)
            cxr_img.add(Text("CXR", font_size=FS_SMALL).move_to(cxr_img))
        
        cxr_label = Text("Vstupní snímek", font_size=FS_CAPTION).next_to(cxr_img, DOWN, buff=0.2)
        cxr_node = Group(cxr_img, cxr_label)

        # ── 2. ResNet-50 ────────────────────────────────────────────────
        resnet_node = make_node("ResNet-50", PURE_CYAN)

        # ── 3. PCA ──────────────────────────────────────────────────────
        pca_node = make_node("PCA", YELLOW, width=1.6)

        # ── 4. VQC ──────────────────────────────────────────────────────
        vqc_node = make_node("VQC", PURPLE, width=1.6)

        # ── 5. Output Classes ───────────────────────────────────────────
        out_norm = Text("NORMAL", font_size=FS_SMALL, color=GREEN_C)
        out_pneu = Text("PNEUMONIA", font_size=FS_SMALL, color=RED_C)
        out_node = VGroup(out_norm, out_pneu).arrange(DOWN, buff=0.6)

        # ── Layout ──────────────────────────────────────────────────────
        pipeline = Group(cxr_node, resnet_node, pca_node, vqc_node, out_node)
        pipeline.arrange(RIGHT, buff=1.0).move_to(DOWN * 0.2)

        # ── Arrows ──────────────────────────────────────────────────────
        def make_arrow(start_mob, end_mob):
            return Arrow(
                start_mob.get_right(), end_mob.get_left(), 
                buff=0.15, color=WHITE, stroke_width=3, max_tip_length_to_length_ratio=0.1
            )

        a1 = make_arrow(cxr_node, resnet_node)
        a2 = make_arrow(resnet_node, pca_node)
        a3 = make_arrow(pca_node, vqc_node)

        # Branching arrows for final output
        a4_norm = Arrow(vqc_node.get_right(), out_norm.get_left(), buff=0.15, color=WHITE, stroke_width=3)
        a4_pneu = Arrow(vqc_node.get_right(), out_pneu.get_left(), buff=0.15, color=WHITE, stroke_width=3)
    

        # ── Animation Sequence ──────────────────────────────────────────
        self.play(Write(title))
        self.next_slide()

        # Step 1: Input
        self.play(FadeIn(cxr_node, shift=RIGHT * 0.3), run_time=0.8)
        self.next_slide()

        # Step 2: Classical feature extraction
        self.play(GrowArrow(a1), run_time=0.6)
        self.play(FadeIn(resnet_node, shift=RIGHT * 0.3), run_time=0.8)
        self.next_slide()

        # Step 3: Dimensionality reduction
        self.play(GrowArrow(a2), run_time=0.6)
        self.play(FadeIn(pca_node, shift=RIGHT * 0.3), run_time=0.8)
        self.next_slide()

        # Step 4: Quantum Classifier
        self.play(GrowArrow(a3), run_time=0.6)
        self.play(FadeIn(vqc_node, shift=RIGHT * 0.3), run_time=0.8)
        self.next_slide()

        # Step 5: Final Classification
        self.play(GrowArrow(a4_norm), GrowArrow(a4_pneu), run_time=0.6)
        self.play(FadeIn(out_node, shift=RIGHT * 0.3), run_time=0.8)
        self.next_slide()
        
        self.wait(1.0)
        self.next_slide()
        
        # ── Clean up ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_16: Kaggle Dataset & Data Split Chart ──────────────────────────
    def slide_16(self):

        # ── 1. Load & Show Kaggle Image ─────────────────────────────────
        try:
            kaggle_img = ImageMobject(IMG_KAGGLE).scale_to_fit_height(5.0)
        except OSError:
            box = Rectangle(width=8, height=5, color=BLUE, fill_opacity=0.1)
            text = Text("Obrázek Kaggle nenalezen\n(kaggle.png)", font_size=FS_HEADING, color=BLUE)
            kaggle_img = VGroup(box, text)

        self.play(FadeIn(kaggle_img), run_time=1.0)
        self.next_slide()

        # Fade out image to make room for the chart
        self.play(FadeOut(kaggle_img), run_time=0.8)

        # ── 2. Create Grouped Bar Chart (Original) ──────────────────────
        chart_title = Text("Počet obrázků podle tříd a splitů", font_size=FS_TITLE, weight=BOLD)
        chart_title.to_edge(UP, buff=0.4)

        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4500, 1000],
            x_length=10,
            y_length=5.5,
            axis_config={"color": WHITE, "include_tip": False},
            y_axis_config={"numbers_to_include": np.arange(0, 4001, 1000)}
        ).shift(DOWN * 0.3)

        # Custom Axis Labels
        x_labels = VGroup(
            Text("train", font_size=FS_BODY).next_to(axes.c2p(1, 0), DOWN, buff=0.2),
            Text("test", font_size=FS_BODY).next_to(axes.c2p(2, 0), DOWN, buff=0.2),
            Text("val", font_size=FS_BODY).next_to(axes.c2p(3, 0), DOWN, buff=0.2),
        )

        y_label = Text("Počet obrázků", font_size=FS_BODY, weight=BOLD).rotate(90 * DEGREES)
        y_label.next_to(axes.y_axis, LEFT, buff=0.3)

        x_label = Text("", font_size=FS_BODY, weight=BOLD)
        x_label.next_to(axes.x_axis, DOWN, buff=0.8)

        # Faint Grid lines (every 500 units)
        grid_lines = VGroup(*[
            DashedLine(
                start=axes.c2p(0, y),
                end=axes.c2p(4, y),
                color=GRAY, stroke_width=1, stroke_opacity=0.3
            ) for y in range(500, 4500, 500)
        ])

        # ── 3. Data & Bars (Original Split) ─────────────────────────────
        splits = [1, 2, 3]
        normal_data = [1341, 234, 8]
        pneu_data   = [3875, 390, 8]

        BAR_WIDTH    = 0.9   # scene-unit width of each bar; fits two per tick without overlap

        def _build_bars(norm_vals, pneu_vals):
            """Return (bars VGroup, bar_labels VGroup) for a dataset."""
            grp_bars   = VGroup()
            grp_labels = VGroup()
            y0 = axes.c2p(0, 0)[1]
            for i, (n_val, p_val) in enumerate(zip(norm_vals, pneu_vals)):
                x = splits[i]
                for val, color, side in [
                    (n_val, COLOR_NORMAL, LEFT),
                    (p_val, COLOR_PNEU,   RIGHT),
                ]:
                    h   = max(axes.c2p(0, val)[1] - y0, 0.02)
                    bar = Rectangle(width=BAR_WIDTH, height=h,
                                    color=color, fill_opacity=0.9, stroke_width=0)
                    bar.move_to(axes.c2p(x, 0) + side * (BAR_WIDTH / 2), aligned_edge=DOWN)
                    lbl = Text(str(val), font_size=FS_SMALL, weight=BOLD).next_to(bar, UP, buff=0.1)
                    grp_bars.add(bar)
                    grp_labels.add(lbl)
            return grp_bars, grp_labels

        bars,     bar_labels     = _build_bars(normal_data, pneu_data)

        # ── 4. Legend ───────────────────────────────────────────────────
        leg_n_box = Square(side_length=0.25, color=COLOR_NORMAL, fill_opacity=0.9, stroke_width=0)
        leg_n_txt = Text("NORMAL", font_size=FS_SMALL)
        leg_n = VGroup(leg_n_box, leg_n_txt).arrange(RIGHT, buff=0.2)

        leg_p_box = Square(side_length=0.25, color=COLOR_PNEU, fill_opacity=0.9, stroke_width=0)
        leg_p_txt = Text("PNEUMONIA", font_size=FS_SMALL)
        leg_p = VGroup(leg_p_box, leg_p_txt).arrange(RIGHT, buff=0.2)

        legend = VGroup(leg_n, leg_p).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.move_to(axes.c2p(3.5, 3800))
        
        legend_bg = SurroundingRectangle(legend, color=BLACK, fill_opacity=0.6, stroke_width=0, buff=0.15)
        legend_group = VGroup(legend_bg, legend)

        # ── 5. Show Original Chart ──────────────────────────────────────
        self.play(Write(chart_title))
        self.play(
            Create(axes), Create(grid_lines), 
            Write(x_labels), Write(y_label), Write(x_label),
            FadeIn(legend_group)
        )
        
        self.play(
            LaggedStart(*[GrowFromEdge(bar, DOWN) for bar in bars], lag_ratio=0.1),
            run_time=1.5, rate_func=smooth
        )
        self.play(Write(bar_labels))

        self.next_slide()

        # ── 6. Reassignment Animation (Train + Val → 80/20 split) ──────
        subtitle = Text("Nové rozdělení (Sloučení Train+Val, split 80/20)", font_size=FS_SUB, color=YELLOW)
        subtitle.next_to(chart_title, DOWN, buff=0.2)

        new_bars, new_bar_labels = _build_bars([1079, 234, 270], [3106, 390, 777])

        # Morph the chart!
        self.play(Write(subtitle))
        
        # Fade out old numbers, stretch the bars, fade in new numbers
        self.play(FadeOut(bar_labels), run_time=0.4)
        self.play(
            *[Transform(bars[i], new_bars[i]) for i in range(len(bars))],
            run_time=1.5, rate_func=smooth
        )
        self.play(FadeIn(new_bar_labels), run_time=0.5)

        self.wait(1.0)
        self.next_slide()
        
        # ── Clean up ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_17: Enhanced Visual Preprocessing ─────────────────────────────
    def slide_17(self):

        # ── 1. Original Image (Initially stretched) ─────────────────────
        try:
            img = ImageMobject(IMG_CXR)
            img.scale_to_fit_height(5.5)
            img.stretch(1.4, dim=0) 
        except OSError:
            img = Rectangle(width=6.0, height=5.0, color=GRAY, fill_opacity=0.5)

        img.move_to(ORIGIN)
        self.play(FadeIn(img))
        self.next_slide()

        # ── 2. Transformation: Resize (with labels) ─────────────────────
        self.play(
            img.animate.stretch(1/1.4, dim=0).scale_to_fit_height(4.5),
            run_time=1.2
        )
        
        # Adding braces and "224" labels back
        b_top = Brace(img, UP, color=YELLOW)
        b_right = Brace(img, RIGHT, color=YELLOW)
        t_top = b_top.get_text("224").set_color(YELLOW)
        t_right = b_right.get_text("224").set_color(YELLOW)

        self.play(
            Create(b_top), Write(t_top),
            Create(b_right), Write(t_right)
        )
        self.next_slide()

        # ── 3. Transformation: ToTensor (Split + Grid) ──────────────────
        # We fade out the image and replace it with 3 colored layers that have a "mesh"
        layer_offset = UP * 0.25 + RIGHT * 0.25
        
        layers = VGroup()
        for i, color in enumerate([BLUE_C, GREEN_C, RED_C]):
            # Create a tinted plane
            plane = Rectangle(width=4.5, height=4.5, color=color, fill_opacity=0.3, stroke_width=2)
            # Add a grid/mesh to represent the "Tensor" structure of numbers
            grid = NumberPlane(
                x_range=[0, 10, 1], y_range=[0, 10, 1],
                x_length=4.5, y_length=4.5,
                background_line_style={"stroke_color": color, "stroke_width": 1, "stroke_opacity": 0.4}
            ).move_to(plane)
            
            layer = VGroup(plane, grid).move_to(img.get_center() + layer_offset * i)
            layers.add(layer)

        self.play(
            FadeOut(img), FadeOut(b_top), FadeOut(t_top), FadeOut(b_right), FadeOut(t_right),
            LaggedStart(*[FadeIn(l, shift=layer_offset*0.5) for l in layers], lag_ratio=0.3),
            run_time=1.5
        )
        self.next_slide()

        # ── 4. Transformation: Normalization (Processing Sweep) ─────────
        # Create a "Scan line" that passes through to represent calculation
        sweep_line = Line(layers.get_left() + LEFT*0.5, layers.get_left() + LEFT*0.5 + UP*5, color=WHITE)
        sweep_line.set_stroke(width=10).set_opacity(0.8)
        sweep_line.set_y(layers.get_center()[1])

        # The animation: Scan line moves across while colors shift
        self.play(
            sweep_line.animate.move_to(layers.get_right() + RIGHT*0.5),
            layers.animate.set_color(PURPLE_A),
            run_time=2,
            rate_func=linear
        )
        self.play(FadeOut(sweep_line))
        
        # Final "Indicator" pulse to show data is ready
        self.play(
            layers.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=0.8
        )
        self.next_slide()

        # ── Cleanup ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_18: ResNet-50 Architecture Visualization ──────────────────────
    def slide_18(self):
        # Set a nice isometric 3D perspective
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES, run_time=1.5)

        title = Text("Resnet-50", font_size=FS_LARGE, weight=BOLD)
        # Fix title to the screen frame so it doesn't rotate in 3D
        self.add_fixed_in_frame_mobjects(title)
        title.to_edge(UP, buff=0.4)

        # ── 1. Create the Convolutional Volumes ─────────────────────────
        def make_volume(size, depth, color, opacity=0.3):
            layers = VGroup()
            # We stack several rectangles to represent the "depth" (channels)
            for i in range(depth):
                rect = Rectangle(
                    width=size, height=size, 
                    color=color, fill_opacity=opacity, stroke_width=1
                )
                rect.shift(OUT * i * 0.15) # Shift along Z-axis
                layers.add(rect)
            return layers

        # Layer 1: Input-like (Large, shallow)
        v1 = make_volume(size=3.0, depth=3, color=BLUE_C).move_to(LEFT * 5)
        
        # Layer 2: Early Conv (Smaller, deeper)
        v2 = make_volume(size=2.2, depth=6, color=PURE_CYAN).move_to(LEFT * 1.5)
        
        # Layer 3: Middle Conv (Even smaller, even deeper)
        v3 = make_volume(size=1.4, depth=10, color=PURPLE_A).move_to(RIGHT * 1.5)
        
        # Layer 4: Final Feature Maps (Smallest, deepest)
        v4 = make_volume(size=0.8, depth=15, color=PINK).move_to(RIGHT * 4.5)

        # ── 2. Connecting Arrows ───────────────────────────────────────
        arrows = VGroup(*[
            Arrow(v1.get_right(), v2.get_left(), color=WHITE, buff=0.2),
            Arrow(v2.get_right(), v3.get_left(), color=WHITE, buff=0.2),
            Arrow(v3.get_right(), v4.get_left(), color=WHITE, buff=0.2),
        ])

        # ── 3. The Signature Residual (Skip) Connection ────────────────
        # A curved arrow jumping over Layer 3
        skip_connection = CurvedArrow(
            v2.get_top() + UP * 0.2, 
            v4.get_top() + UP * 0.2, 
            angle=-TAU / 4, 
            color=YELLOW, 
            stroke_width=6
        )
        
        # Plus sign at the end of the skip connection
        plus_sign = MathTex("+", color=YELLOW, font_size=FS_TITLE).next_to(v4, UP, buff=0.5)

        # ── Animation Sequence ──────────────────────────────────────────
        self.play(Write(title))
        self.next_slide()

        # Reveal volumes and standard connections
        self.play(
            LaggedStart(
                FadeIn(v1, shift=IN),
                GrowArrow(arrows[0]),
                FadeIn(v2, shift=IN),
                GrowArrow(arrows[1]),
                FadeIn(v3, shift=IN),
                GrowArrow(arrows[2]),
                FadeIn(v4, shift=IN),
                lag_ratio=0.4
            ),
            run_time=3.0
        )
        self.next_slide()

        # Highlight the Skip Connection (The "Res" in ResNet)
        self.play(
            Create(skip_connection),
            FadeIn(plus_sign, shift=DOWN),
            v2.animate.set_color(YELLOW),
            v4.animate.set_color(YELLOW),
            run_time=1.5
        )
        self.play(Indicate(skip_connection, scale_factor=1.1, color=WHITE))
        self.next_slide()

        # Optional: A quick camera rotation to show off the 3D depth

        # ── Clean up ────────────────────────────────────────────────────
        # Reset to 2-D first, then fade so the next slide starts clean.
        self.move_camera(phi=0, theta=-90 * DEGREES, run_time=1.0)
        _fade_all(self)

    # ── slide_19: PCA (Principal Component Analysis) Visualization ───────────
    def slide_19(self):

        title = Text("PCA: Redukce dimenzionality", font_size=FS_SLIDE, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── 1. Generate Correlated Random Data ──────────────────────────
        np.random.seed(42)
        n_points = 60
        # Create a correlated cloud of points
        x = np.random.normal(0, 1.5, n_points)
        y = 0.6 * x + np.random.normal(0, 0.5, n_points)
        points_data = np.vstack([x, y]).T

        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-4, 4, 1],
            x_length=8,
            y_length=6,
            axis_config={"color": WHITE, "include_tip": False}
        ).shift(DOWN * 0.3)

        dots = VGroup(*[
            Dot(axes.c2p(p[0], p[1]), radius=0.06, color=BLUE_B, fill_opacity=0.7)
            for p in points_data
        ])

        self.play(Create(axes), FadeIn(dots, lag_ratio=0.05), run_time=2)
        self.next_slide()

        # ── 2. Calculate Principal Components (Math) ────────────────────
        # Center the data
        mean_vec = np.mean(points_data, axis=0)
        centered_data = points_data - mean_vec
        # Covariance matrix and Eigenvectors
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # ── 3. Draw Principal Components ────────────────────────────────
        # PC1 (Direction of max variance)
        pc1_dir = eigenvectors[:, 0]
        pc1_arrow = Arrow(
            axes.c2p(0, 0), 
            axes.c2p(pc1_dir[0] * 3, pc1_dir[1] * 3), 
            color=YELLOW, buff=0, stroke_width=6
        )
        pc1_label = MathTex(r"PC_1", color=YELLOW).next_to(pc1_arrow.get_end(), UR, buff=0.1)

        # PC2 (Orthogonal to PC1)
        pc2_dir = eigenvectors[:, 1]
        pc2_arrow = Arrow(
            axes.c2p(0, 0), 
            axes.c2p(pc2_dir[0] * 1.5, pc2_dir[1] * 1.5), 
            color=PURPLE_A, buff=0, stroke_width=6
        )
        pc2_label = MathTex(r"PC_2", color=PURPLE_A).next_to(pc2_arrow.get_end(), UL, buff=0.1)

        self.play(GrowArrow(pc1_arrow), Write(pc1_label))
        self.play(GrowArrow(pc2_arrow), Write(pc2_label))
        self.next_slide()

        # ── 4. Projection onto PC1 (Dimensionality Reduction) ───────────
        # Project points onto the PC1 line: (p dot v) * v
        projected_points = []
        projection_lines = VGroup()

        for p in points_data:
            proj_val = np.dot(p, pc1_dir)
            proj_point = proj_val * pc1_dir
            proj_coord = axes.c2p(proj_point[0], proj_point[1])
            
            projected_points.append(proj_coord)
            # Faint dashed lines showing the projection path
            line = DashedLine(
                axes.c2p(p[0], p[1]), 
                proj_coord, 
                color=GRAY, stroke_width=1, stroke_opacity=0.5
            )
            projection_lines.add(line)

        proj_dots = VGroup(*[
            Dot(p, radius=0.05, color=YELLOW) for p in projected_points
        ])

        self.play(Create(projection_lines), FadeIn(proj_dots), run_time=1.5)
        self.next_slide()

        # Fade out original dots and PC2 to show the "Reduced" 1D data
        self.play(
            FadeOut(dots),
            FadeOut(pc2_arrow),
            FadeOut(pc2_label),
            FadeOut(projection_lines),
            proj_dots.animate.set_color(PURE_CYAN).scale(1.2),
            run_time=1.0
        )

        self.next_slide()

        # ── Clean up ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_20: VQC (Variational Quantum Classifier) ──────────────────────
    def slide_20(self):

        title = Text("Variational Quantum Classifier (VQC)", font_size=FS_SLIDE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── 1. Load Circuit as PNG ──────────────────────────────────────
        try:
            # Přímé načtení rastrového obrázku
            circuit = ImageMobject(IMG_CIRCUIT)
            circuit.scale_to_fit_width(9.0)
        except OSError:
            # Fallback placeholder, kdyby soubor nebyl nalezen
            box = Rectangle(width=9.0, height=3.0, color=PURPLE_A, fill_opacity=0.1)
            circuit_txt = Text("Chybí circuit.png", font_size=FS_BODY)
            # Použijeme Group (místo VGroup), protože ImageMobject není vektor
            circuit = Group(box, circuit_txt.move_to(box))
            
        circuit.move_to(ORIGIN + UP * 0.5)

        self.play(FadeIn(circuit), run_time=1.5)
        self.next_slide()

        # ── 2. Explain the 3 phases of VQC ──────────────────────────────
        labels = VGroup(
            Text("1. Kódování dat", font_size=FS_BASE, color=PURE_CYAN, weight=BOLD),
            Text("2. Ansatz (Váhy)", font_size=FS_BASE, color=YELLOW, weight=BOLD),
            Text("3. Měření", font_size=FS_BASE, color=RED_C, weight=BOLD)
        ).arrange(RIGHT, buff=1.0).to_edge(DOWN, buff=1.2)

        descs = VGroup(
            Text("Klasická data → Kvantový stav", font_size=FS_SMALL, color=WHITE),
            Text("Parametrizované rotace", font_size=FS_SMALL, color=WHITE),
            Text("Kolaps stavu → Výstup", font_size=FS_SMALL, color=WHITE)
        )
        for i, desc in enumerate(descs):
            desc.next_to(labels[i], DOWN, buff=0.2)

        # Výpočet rozměrů pro highlight box
        w = circuit.width
        h = circuit.height + 0.5
        
        hl_box = Rectangle(width=w/3, height=h, color=PURE_CYAN, fill_opacity=0.15, stroke_width=3)
        # Umístění nad levou třetinu (Encoding)
        hl_box.move_to(circuit.get_left() + RIGHT * (w/6))

        # Ukázat Fázi 1: Encoding
        self.play(
            FadeIn(labels[0], shift=UP * 0.2), 
            FadeIn(descs[0]),
            Create(hl_box)
        )
        self.next_slide()

        # Ukázat Fázi 2: Ansatz (Přesun doprostřed)
        self.play(
            hl_box.animate.move_to(circuit.get_center()).set_color(YELLOW),
            FadeIn(labels[1], shift=UP * 0.2),
            FadeIn(descs[1]),
            run_time=1.0
        )
        self.next_slide()

        # Ukázat Fázi 3: Měření (Přesun doprava)
        self.play(
            hl_box.animate.move_to(circuit.get_right() + LEFT * (w/6)).set_color(RED_C),
            FadeIn(labels[2], shift=UP * 0.2),
            FadeIn(descs[2]),
            run_time=1.0
        )
        self.next_slide()

        # ── Cleanup ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_21: Metody kódování dat (Data Encoding) ───────────────────────
    def slide_21(self):

        # ── 1. Zobrazení všech tří možností (Horizontálně, bez čísel, větší) ─
        opt1 = Text("Amplitude Encoding", font_size=FS_HEADING, color=BLUE_C)
        opt3 = Text("Angle Encoding", font_size=FS_HEADING, color=GREEN_C)
        
        options = VGroup(opt1, opt3).arrange(RIGHT, buff=0.8)
        options.move_to(ORIGIN)
        
        self.play(LaggedStart(
            FadeIn(opt1, shift=UP*0.2),
            FadeIn(opt3, shift=UP*0.2),
            lag_ratio=0.3
        ), run_time=2.0)
        self.next_slide()

        # Zmizení úvodního menu
        self.play(FadeOut(options), run_time=0.8)

        # ── 2. Animace: Amplitude Encoding ──────────────────────────────
        amp_title = opt1.copy().scale(1.2).to_edge(UP, buff=0.5)
        
        vec_math = MathTex(
            r"\vec{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix}",
            color=WHITE
        )
        arrow_amp = Arrow(LEFT, RIGHT, color=BLUE_C)
        state_math = MathTex(
            r"|\psi\rangle = x_1|00\rangle + x_2|01\rangle + x_3|10\rangle + x_4|11\rangle",
            color=BLUE_C
        )
        
        amp_group = VGroup(vec_math, arrow_amp, state_math).arrange(RIGHT, buff=0.5)
        
        self.play(FadeIn(amp_title))
        self.play(Write(vec_math))
        self.play(GrowArrow(arrow_amp))
        self.play(Write(state_math))
        self.next_slide()
        
        self.play(FadeOut(amp_group), FadeOut(amp_title))

        # ── 4. Animace: Angle Encoding ──────────────────────────────────
        ang_title = opt3.copy().scale(1.2).to_edge(UP, buff=0.5)
        
        # Vizualizace Angle Encoding jako rotace vektoru v kružnici (2D řez sférou)
        circle = Circle(radius=1.8, color=WHITE, stroke_width=2)
        y_axis = Line(UP * 2.0, DOWN * 2.0, color=GRAY)
        x_axis = Line(LEFT * 2.0, RIGHT * 2.0, color=GRAY)
        
        # Výchozí stav |0> (ukazuje nahoru)
        vec_line = Line(ORIGIN, UP * 1.8, color=GREEN_C, stroke_width=5)
        state_0 = MathTex(r"|0\rangle").next_to(y_axis, UP, buff=0.1)
        state_1 = MathTex(r"|1\rangle").next_to(y_axis, DOWN, buff=0.1)
        
        ang_group = VGroup(circle, y_axis, x_axis, state_0, state_1)
        
        self.play(FadeIn(ang_title))
        self.play(Create(ang_group), GrowFromCenter(vec_line))
        self.next_slide()

        # Animace rotace o úhel x_i
        target_angle = -PI / 3  # Simulujeme rotaci o cca 60 stupňů doprava
        new_vec = Line(ORIGIN, UP * 1.8, color=GREEN_C, stroke_width=5)
        new_vec.rotate(target_angle, about_point=ORIGIN)
        
        # Vykreslení oblouku rotace
        angle_arc = Arc(radius=0.6, start_angle=PI/2, angle=target_angle, color=YELLOW, stroke_width=4)
        
        # PUSHED TEXT TO THE RIGHT: Odsunuto bezpečně mimo osy
        theta_txt = MathTex(r"\theta = x_i", color=YELLOW).next_to(angle_arc, RIGHT, buff=0.8).shift(UP * 0.3)

        self.play(
            Transform(vec_line, new_vec),
            Create(angle_arc),
            Write(theta_txt),
            run_time=1.5,
            rate_func=smooth
        )
        self.play(Indicate(theta_txt, color=WHITE))
        self.next_slide()

        # ── Cleanup ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_22: Ansatz a Strongly Entangling Layers ────────────────────────
    def slide_22(self):

        title = Text("Ansatz (Parametrizovaný obvod)", font_size=FS_SLIDE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── 1. Obecný princip Ansatzu (Big Picture) ─────────────────────
        # Kvantový blok
        q_box = Rectangle(width=3.5, height=2.5, color=PURPLE_A, fill_opacity=0.2)
        q_text = MathTex(r"U(\vec{\theta})", font_size=FS_LARGE, color=WHITE).move_to(q_box)
        q_label = Text("Kvantový obvod", font_size=FS_SMALL, color=PURPLE_A).next_to(q_box, UP)
        quantum_block = VGroup(q_box, q_text, q_label)

        # Šipky vstupu a výstupu
        arr_in = Arrow(LEFT * 4.5, q_box.get_left(), color=WHITE)
        arr_out = Arrow(q_box.get_right(), RIGHT * 4.5, color=WHITE)
        psi_in = MathTex(r"|\psi_{in}\rangle").next_to(arr_in, UP, buff=0.1)
        psi_out = MathTex(r"|\psi_{out}\rangle").next_to(arr_out, UP, buff=0.1)

        # Klasický optimalizátor (Smyčka)
        opt_box = Rectangle(width=3.5, height=1.2, color=YELLOW, fill_opacity=0.2)
        opt_box.move_to(DOWN * 2.5)
        opt_text = Text("Klasický optimalizátor\n(Upravuje váhy)", font_size=FS_SMALL, color=YELLOW).move_to(opt_box)
        optimizer_block = VGroup(opt_box, opt_text)

        # Propojení smyčky
        loop_down = CurvedArrow(arr_out.get_end() + DOWN*0.2, opt_box.get_right() + UP*0.2, angle=-TAU/4, color=GRAY)
        loop_up = CurvedArrow(opt_box.get_left() + UP*0.2, q_box.get_bottom() + DOWN*0.2, angle=-TAU/4, color=YELLOW)
        theta_update = MathTex(r"\text{Nové } \vec{\theta}", color=YELLOW, font_size=FS_SUB).next_to(loop_up, LEFT, buff=0.2)

        general_group = VGroup(
            quantum_block, arr_in, arr_out, psi_in, psi_out, 
            optimizer_block, loop_down, loop_up, theta_update
        )

        self.play(FadeIn(quantum_block), GrowArrow(arr_in), Write(psi_in), GrowArrow(arr_out), Write(psi_out))
        self.play(FadeIn(optimizer_block), Create(loop_down))
        self.play(Create(loop_up), Write(theta_update))
        self.next_slide()

        # Úklid první části
        self.play(FadeOut(general_group), run_time=0.8)

        # ── 2. Strongly Entangling Layers (PennyLane) ───────────────────
        subtitle = Text("", font_size=FS_BASE, color=PURE_CYAN)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(subtitle))

        # Pomocné funkce pro kreslení obvodu
        y_coords = [1.0, 0.0, -1.0] # Y pozice pro 3 qubity
        
        # Kreslení "drátů" (qubitů)
        wires = VGroup(*[
            Line(LEFT * 5, RIGHT * 5, color=GRAY, stroke_width=2).set_y(y)
            for y in y_coords
        ])
        q_labels = VGroup(*[
            MathTex(f"q_{i}").next_to(wires[i], LEFT, buff=0.2) for i in range(3)
        ])

        self.play(Create(wires), Write(q_labels), run_time=1.0)

        # KROK A: Rotace (Parametry)
        rot_boxes = VGroup()
        for i in range(3):
            # Rot(theta) box
            box = Rectangle(width=1.8, height=0.7, color=BLUE_C, fill_opacity=0.8)
            box.move_to(LEFT * 2.5 + UP * y_coords[i])
            txt = MathTex(r"Rot(\vec{\theta}_{" + str(i) + r"})", font_size=FS_BODY, color=WHITE).move_to(box)
            rot_boxes.add(VGroup(box, txt))
        
        rot_desc = Text("1. Rotace: 3 parametry na qubit", font_size=FS_BODY, color=BLUE_C).next_to(wires, DOWN, buff=1.0)

        self.play(LaggedStart(*[FadeIn(rb, shift=DOWN*0.2) for rb in rot_boxes], lag_ratio=0.2))
        self.play(Write(rot_desc))
        self.next_slide()

        # KROK B: CNOT hradla (Entanglement)
        # Helper pro nakreslení CNOT hradla
        def make_cnot(q_ctrl, q_targ, x_pos):
            ctrl_dot = Dot(radius=0.1, color=RED_C).move_to(RIGHT * x_pos + UP * y_coords[q_ctrl])
            targ_circle = Circle(radius=0.15, color=RED_C, stroke_width=3).move_to(RIGHT * x_pos + UP * y_coords[q_targ])
            targ_cross = VGroup(
                Line(targ_circle.get_top(), targ_circle.get_bottom(), color=RED_C),
                Line(targ_circle.get_left(), targ_circle.get_right(), color=RED_C)
            )
            line = Line(ctrl_dot.get_center(), targ_circle.get_center(), color=RED_C, stroke_width=3)
            return VGroup(line, ctrl_dot, targ_circle, targ_cross)

        cnots = VGroup(
            make_cnot(0, 1, x_pos=0.0), # q0 -> q1
            make_cnot(1, 2, x_pos=1.5), # q1 -> q2
            make_cnot(2, 0, x_pos=3.0)  # q2 -> q0 (Uzavírá kruh)
        )

        ent_desc = Text("2. Silné provázání: CNOT hradla do kruhu", font_size=FS_BODY, color=RED_C).next_to(rot_desc, DOWN, buff=0.2)

        self.play(LaggedStart(*[GrowFromCenter(c) for c in cnots], lag_ratio=0.4))
        self.play(Write(ent_desc))
        self.next_slide()

        # KROK C: Zvýraznění celé vrstvy (Opakování)
        layer_box = DashedVMobject(
            Rectangle(width=8.0, height=3.2, color=YELLOW, stroke_width=2),
            num_dashes=30
        ).move_to(RIGHT * 0.25)
        
        layer_label = MathTex(r"\times L \text{ vrstev}", color=YELLOW, font_size=FS_BASE).next_to(layer_box, UP, buff=0.2)

        self.play(Create(layer_box), Write(layer_label))
        self.next_slide()

        # ── Cleanup ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_23: Měření (Single Qubit Expectation Value) ───────────────────
    def slide_23(self):

        title = Text("Měření", font_size=FS_SLIDE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── 1. Ukázka PennyLane kódu (Bulletproof verze) ────────────────
        code_str = "return qml.expval(qml.PauliZ(0))"
        
        # Obyčejný text obarvený jako kód
        code_text = Text(code_str, font_size=FS_SUB, color=PURE_CYAN)
        
        # Pěkný rámeček kolem textu simulující okno editoru
        code_bg = SurroundingRectangle(
            code_text, 
            color=GRAY, 
            fill_color=BLACK, 
            fill_opacity=0.6, 
            corner_radius=0.1,
            buff=0.3
        )
        
        code_block = VGroup(code_bg, code_text).next_to(title, DOWN, buff=0.3)

        self.play(FadeIn(code_block))
        self.next_slide()

        # ── 2. Helper: Symbol měření ────────────────────────────────────
        def make_measure_box():
            box = Rectangle(width=0.8, height=0.8, color=WHITE, fill_color=BLACK, fill_opacity=1)
            arc = Arc(radius=0.25, start_angle=0, angle=PI, color=WHITE)
            arc.move_to(box).shift(DOWN * 0.1)
            arrow = Arrow(
                arc.get_center() + DOWN*0.05, 
                arc.get_center() + UP*0.25 + RIGHT*0.15, 
                buff=0, color=RED_C, stroke_width=2, max_tip_length_to_length_ratio=0.2
            )
            return VGroup(box, arc, arrow)

        # ── 3. Kvantový registr (Měříme jen q_0) ────────────────────────
        n_qubits = 4
        wires = VGroup()
        
        for i in range(n_qubits):
            # Posuneme obvod trochu nahoru, abychom měli dole místo na osu
            y_pos = 1.0 - i * 0.8
            wire = Line(LEFT * 5, RIGHT * 2, color=GRAY).set_y(y_pos)
            wires.add(wire)

        q_labels = VGroup(*[
            MathTex(f"q_{i}").next_to(wires[i], LEFT, buff=0.2) for i in range(n_qubits)
        ])

        # Krabička měření POUZE na q_0
        m_box = make_measure_box().move_to(RIGHT * 1.5 + UP * 1.0)
        
        self.play(Create(wires), Write(q_labels), run_time=1.0)
        self.play(FadeIn(m_box, shift=LEFT*0.5))
        self.next_slide()

        # ── 4. Výstup: Jedno skalární číslo [-1, 1] ─────────────────────
        exp_math = MathTex(
            r"\langle Z_0 \rangle \in [-1, 1]", 
            font_size=FS_HEADING, color=YELLOW
        ).next_to(m_box, RIGHT, buff=0.5)

        self.play(Write(exp_math))
        self.next_slide()

        # ── 5. Klasifikace na číselné ose (Number Line) ─────────────────
        # Osa od -1 do 1
        number_line = NumberLine(
            x_range=[-1, 1, 0.5],
            length=8,
            color=WHITE,
            include_numbers=True,
            numbers_to_include=[-1, 0, 1],
            font_size=FS_BODY
        ).shift(DOWN * 2.5)

        # Barevné podbarvení oblastí (Třídy)
        zone_normal = Rectangle(width=4, height=0.5, color=GREEN_C, fill_opacity=0.3, stroke_width=0)
        zone_normal.next_to(number_line.n2p(0), LEFT, buff=0)
        
        zone_pneu = Rectangle(width=4, height=0.5, color=RED_C, fill_opacity=0.3, stroke_width=0)
        zone_pneu.next_to(number_line.n2p(0), RIGHT, buff=0)

        label_norm = Text("NORMAL", font_size=FS_SMALL, color=GREEN_C).next_to(zone_normal, UP, buff=0.1)
        label_pneu = Text("PNEUMONIA", font_size=FS_SMALL, color=RED_C).next_to(zone_pneu, UP, buff=0.1)

        self.play(
            Create(number_line),
            FadeIn(zone_normal), FadeIn(zone_pneu),
            Write(label_norm), Write(label_pneu)
        )
        self.next_slide()

        # ── 6. Animace vyhodnocení konkrétního snímku ───────────────────
        # Naměřená hodnota například 0.65
        measured_val = 0.65
        
        # Tečka reprezentující výstup, vyletí z měřící krabičky
        val_dot = Dot(color=YELLOW, radius=0.1).move_to(m_box.get_right())
        val_text = Text("0.65", font_size=FS_BODY, color=YELLOW).next_to(val_dot, UP, buff=0.1)
        
        # Cílová pozice na ose
        target_pos = number_line.n2p(measured_val)

        self.play(FadeIn(val_dot), Write(val_text))
        
        # Křivka, po které tečka poletí na osu
        path = CurvedArrow(val_dot.get_center(), target_pos + UP*0.1, angle=-PI/4, color=YELLOW)
        
        self.play(
            MoveAlongPath(val_dot, path),
            val_text.animate.next_to(target_pos, UP, buff=0.2),
            run_time=1.5,
            rate_func=smooth
        )
        self.play(FadeOut(path))
        
        # Flash vítězné třídy
        self.play(
            label_pneu.animate.scale(1.5).set_color(RED),
            zone_pneu.animate.set_fill(opacity=0.6),
            label_norm.animate.set_opacity(0.3),
            zone_normal.animate.set_fill(opacity=0.1),
            run_time=1.0
        )
        self.next_slide()

        # ── Cleanup ────────────────────────────────────────────────────
        _fade_all(self)

    # ── slide_24: Výsledky modelu (Model Performance) ───────────────────────
    def slide_24(self):

        # Trénovací data jsou definována jako konstanty na začátku souboru
        # (TRAIN_LOSS, VAL_LOSS, FINAL_ACCURACY atd.) – upravte je tam.
        train_loss        = TRAIN_LOSS
        val_loss          = VAL_LOSS
        final_accuracy    = FINAL_ACCURACY
        final_sensitivity = FINAL_SENSITIVITY
        final_specificity = FINAL_SPECIFICITY

        title = Text("Výsledky kvantového modelu", font_size=FS_SLIDE, weight=BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # ── 1. Automatické škálování grafu Loss ─────────────────────────
        epochs = len(train_loss)
        x_vals = list(range(1, epochs + 1))
        
        min_y = max(0.0, round(min(min(train_loss), min(val_loss)) - 0.1, 1))
        max_y = round(max(max(train_loss), max(val_loss)) + 0.1, 1)
        
        x_step = max(1, epochs // 5)
        y_step = max(0.05, round((max_y - min_y) / 5, 2))

        graf_title = Text("Trénovací vs. Validační Loss", font_size=FS_BODY, color=WHITE)
        graf_title.move_to(LEFT * 3 + UP * 1.8)

        axes = Axes(
            x_range=[0, epochs, x_step],
            y_range=[min_y, max_y, y_step],
            x_length=5,
            y_length=3.5,
            axis_config={"color": WHITE, "include_tip": False}
        ).move_to(LEFT * 3 + DOWN * 0.5)

        x_label = axes.get_x_axis_label("Epocha", edge=DOWN, direction=DOWN, buff=0.4).scale(0.6)
        y_label = axes.get_y_axis_label("Loss", edge=LEFT, direction=LEFT, buff=0.4).scale(0.6)

        train_curve = axes.plot_line_graph(x_values=x_vals, y_values=train_loss, line_color=PURE_CYAN, add_vertex_dots=(epochs <= 20))
        val_curve = axes.plot_line_graph(x_values=x_vals, y_values=val_loss, line_color=YELLOW, add_vertex_dots=(epochs <= 20))

        # Legenda ke grafu
        leg_train = VGroup(Line(color=PURE_CYAN, stroke_width=4).set_length(0.5), Text("Train", font_size=FS_SMALL)).arrange(RIGHT)
        leg_val = VGroup(Line(color=YELLOW, stroke_width=4).set_length(0.5), Text("Validation", font_size=FS_SMALL)).arrange(RIGHT)
        legend = VGroup(leg_train, leg_val).arrange(DOWN, aligned_edge=LEFT).next_to(axes, UR, buff=0.1)

        self.play(FadeIn(graf_title), Create(axes), Write(x_label), Write(y_label))
        self.play(Create(train_curve), FadeIn(leg_train))
        self.play(Create(val_curve), FadeIn(leg_val))
        self.next_slide()

        # ── 2. Finální Klasifikační Metriky ─────────────────────────────
        metrics_title = Text("Výkon na Testovacím setu", font_size=FS_BODY, color=WHITE)
        metrics_title.move_to(RIGHT * 3.5 + UP * 1.8)
        self.play(FadeIn(metrics_title))

        def make_metric_bar(name, value, color, y_offset):
            label = Text(name, font_size=FS_SMALL).move_to(RIGHT * 1.5 + UP * y_offset).align_to(RIGHT * 1.5, LEFT)
            val_text = Text(f"{int(value*100)} %", font_size=FS_BODY, weight=BOLD, color=color)
            val_text.next_to(label, RIGHT, buff=2.5).align_to(RIGHT * 5.5, RIGHT)
            
            bg_bar = Rectangle(width=4.0, height=0.2, color=GRAY, fill_color=GRAY, fill_opacity=0.3, stroke_width=0)
            bg_bar.next_to(label, DOWN, buff=0.1, aligned_edge=LEFT)
            
            fg_bar = Rectangle(width=4.0 * value, height=0.2, color=color, fill_color=color, fill_opacity=0.9, stroke_width=0)
            fg_bar.align_to(bg_bar, LEFT).align_to(bg_bar, UP)
            
            return VGroup(label, val_text, bg_bar, fg_bar), fg_bar

        metrics_data = [
            ("Accuracy", final_accuracy, BLUE_C, 0.5),
            ("Sensitivity", final_sensitivity, RED_C, -0.7),
            ("Specificity", final_specificity, GREEN_C, -1.9)
        ]

        metric_groups = VGroup()
        for name, val, color, y_off in metrics_data:
            group, fg = make_metric_bar(name, val, color, y_off)
            metric_groups.add(group)
            
            self.play(FadeIn(group[:-1]), run_time=0.5) 
            self.play(GrowFromEdge(fg, LEFT), run_time=0.8) 

        self.next_slide()

        # ── 3. Zvýraznění Senzitivity ───────────────────────────────────
        box = SurroundingRectangle(metric_groups[1], color=YELLOW, buff=0.2, stroke_width=3)
        note = Text("", font_size=FS_SMALL, color=YELLOW).next_to(box, RIGHT, buff=0.2)

        self.play(Create(box), Write(note))
        self.next_slide()

        # ── Cleanup ────────────────────────────────────────────────────
        _fade_all(self)
            
    # ── last_slide: hybrid QCNN architecture (closing slide) ────────────────
    def last_slide(self):

        CLASSICAL_SIZES = [5, 7, 7, 5]
        QUANTUM_SIZES   = [4, 2]
        NODE_RADIUS     = 0.15
        X_SPACING       = 2.0
        Y_SPACING       = 0.85
        NET_SHIFT_Y     = -0.5   # push network down to clear the title

        title = Text("Hybridní QCNN", font_size=FS_TITLE, color=WHITE).to_edge(UP, buff=0.5)
        total_layers = len(CLASSICAL_SIZES) + len(QUANTUM_SIZES)
        start_x      = -((total_layers - 1) * X_SPACING) / 2

        # ---- Classical nodes & edges ------------------------------------
        classical_nodes = VGroup()
        for i, size in enumerate(CLASSICAL_SIZES):
            layer = VGroup(*[
                Circle(radius=NODE_RADIUS, color=PURE_CYAN, fill_opacity=0.8)
                .move_to([start_x + i * X_SPACING,
                          (j - (size - 1) / 2) * Y_SPACING + NET_SHIFT_Y, 0])
                for j in range(size)
            ])
            classical_nodes.add(layer)

        classical_edges = VGroup(*[
            Line(n1.get_center(), n2.get_center(),
                 stroke_width=3.5, stroke_opacity=0.6, color=PURE_CYAN)
            for i in range(len(CLASSICAL_SIZES) - 1)
            for n1 in classical_nodes[i]
            for n2 in classical_nodes[i + 1]
        ])

        # ---- Quantum nodes & edges --------------------------------------
        q_start_x     = start_x + len(CLASSICAL_SIZES) * X_SPACING
        quantum_nodes = VGroup()
        for i, size in enumerate(QUANTUM_SIZES):
            layer = VGroup(*[
                VGroup(
                    Circle(radius=NODE_RADIUS * 1.3, color=PURPLE, fill_opacity=0.3),
                    Dot(color=PINK, radius=0.04),
                ).move_to([q_start_x + i * X_SPACING,
                           (j - (size - 1) / 2) * Y_SPACING + NET_SHIFT_Y, 0])
                for j in range(size)
            ])
            quantum_nodes.add(layer)

        quantum_edges = VGroup(*[
            DashedLine(n1.get_center(), n2.get_center(),
                       stroke_width=3.0, stroke_opacity=0.8, color=PURPLE_A)
            for i in range(len(QUANTUM_SIZES) - 1)
            for n1 in quantum_nodes[i]
            for n2 in quantum_nodes[i + 1]
        ])

        bridge_edges = VGroup(*[
            Line(n1.get_center(), n2.get_center(),
                 stroke_width=2.0, stroke_opacity=0.5, color=WHITE)
            for n1 in classical_nodes[-1]
            for n2 in quantum_nodes[0]
        ])

        # ---- QR codes ---------------------------------------------------
        try:
            repo_qr = (ImageMobject(IMG_REPO_QR)
                       .set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
                       .set_height(1.5).to_corner(DL, buff=0.15))
            repo_group = Group(repo_qr,
                               Text("Repozitář", font_size=FS_SMALL).next_to(repo_qr, UP, buff=0.2))

            docs_qr = (ImageMobject(IMG_DOCS_QR)
                       .set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
                       .set_height(1.5).to_corner(DR, buff=0.15))
            docs_group = Group(docs_qr,
                               Text("Dokumentace", font_size=FS_SMALL).next_to(docs_qr, UP, buff=0.2))
        except OSError:
            repo_group = Text("(QR kód chybí)", font_size=FS_SMALL, color=RED).to_corner(DL)
            docs_group = Text("(QR kód chybí)", font_size=FS_SMALL, color=RED).to_corner(DR)

        # ---- Reveal -----------------------------------------------------
        entire_network = VGroup(classical_edges, quantum_edges, bridge_edges,
                                classical_nodes, quantum_nodes)
        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(entire_network, repo_group, docs_group), run_time=1.5)

        # ---- Looping activation pulse -----------------------------------
        # Colours and scale factors are sampled fresh inside the loop body
        # so each loop iteration looks different (not frozen at build time).
        BRIGHTNESS = [YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D, YELLOW_E, WHITE]

        self.next_slide(loop=True)

        def _make_layer_anims():
            anims = []
            for layer in classical_nodes:
                anims.append(AnimationGroup(*[
                    Indicate(node,
                             color=random.choice(BRIGHTNESS),
                             scale_factor=random.uniform(1.1, 1.5),
                             run_time=0.6)
                    for node in layer
                ]))
            for layer in quantum_nodes:
                anims.append(AnimationGroup(*[
                    Indicate(node[0],
                             color=random.choice(BRIGHTNESS),
                             scale_factor=random.uniform(1.1, 1.5),
                             run_time=0.6)
                    for node in layer
                ]))
            return anims

        self.play(LaggedStart(*_make_layer_anims(), lag_ratio=0.5))
        self.wait(0.5)

        # ---- Graceful exit ----------------------------------------------
        self.next_slide()
        self.play(
            FadeOut(entire_network),
            FadeOut(title),
            FadeOut(repo_group),
            FadeOut(docs_group),
            run_time=1.0,
    )