"""
SOČ Prezentace — Hybridní model pro detekci pneumonie
====================================================
Manim Slides presentation for Středoškolská odborná činnost 2026.

Usage:
    Render & present:
        python3 presentation.py render SocPresentation    # low quality
        python3 presentation.py render-m SocPresentation  # medium quality
        python3 presentation.py present SocPresentation

    HTML export (after render):
        python3 presentation.py html SocPresentation soc_toggle.html
        python3 presentation.py html BlochSpherePresentation bloch_sphere.html

    All in one:
        python3 presentation.py all                      # render + HTML toggle
        python3 presentation.py all-bloch                # render + HTML Bloch sphere

    Present Bloch sphere:
        python3 presentation.py present-bloch
"""

import os
import re
import sys
import subprocess
import json as json_mod
import shutil

from manim_slides import Slide, ThreeDSlide
from manim import *

# ═══════════════════════════════════════════════════════════════════════════════
# MONKEY-PATCH: Replace broken PyAV concat with ffmpeg CLI
# PyAV 17.x is incompatible with ffmpeg 8.x on this system.
# ═══════════════════════════════════════════════════════════════════════════════

from pathlib import Path
import manim.scene.scene_file_writer as _sfw
import logging as _logging

_sfw_logger = _logging.getLogger("manim.scene.scene_file_writer")


def _patched_combine_files(
    self,
    input_files: list[str],
    output_file,
    create_gif: bool = False,
    includes_sound: bool = False,
) -> None:
    """Replacement for SceneFileWriter.combine_files using ffmpeg CLI."""
    file_list = self.partial_movie_directory / "partial_movie_file_list.txt"
    _sfw_logger.debug(
        f"Partial movie files to combine ({len(input_files)} files): %(p)s",
        {"p": input_files[:5]},
    )
    # Write concat list
    with file_list.open("w", encoding="utf-8") as fp:
        fp.write("# This file is used internally by FFMPEG.\n")
        for pf_path in input_files:
            pf_path_str = Path(pf_path).as_posix()
            fp.write(f"file 'file:{pf_path_str}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(file_list),
        "-c", "copy",
    ]
    if not includes_sound:
        cmd.extend(["-an"])
    cmd.append(str(output_file))

    _sfw_logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _sfw_logger.error(f"ffmpeg stderr: {result.stderr}")
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")


def _patched_combine_to_movie(self) -> None:
    """Replacement for SceneFileWriter.combine_to_movie using ffmpeg CLI."""
    partial_movie_files = [el for el in self.partial_movie_files if el is not None]
    movie_file_path = self.movie_file_path
    if _sfw.is_gif_format():
        movie_file_path = self.gif_file_path

    if len(partial_movie_files) == 0:
        _sfw_logger.info("No animations are contained in this scene.")
        return

    _sfw_logger.info("Combining to Movie file.")
    self.combine_files(
        partial_movie_files,
        movie_file_path,
        _sfw.is_gif_format(),
        self.includes_sound,
    )

    # handle sound
    if self.includes_sound and config.format != "gif":
        sound_file_path = movie_file_path.with_suffix(".wav")
        self.add_audio_segment(_sfw.AudioSegment.silent(0))
        self.audio_segment.export(
            sound_file_path,
            format="wav",
            bitrate="312k",
        )
        if config.movie_file_extension == ".webm":
            ogg_sound_file_path = sound_file_path.with_suffix(".ogg")
            _sfw.convert_audio(sound_file_path, ogg_sound_file_path, "libvorbis")
            sound_file_path = ogg_sound_file_path
        elif config.movie_file_extension == ".mp4":
            aac_sound_file_path = sound_file_path.with_suffix(".aac")
            _sfw.convert_audio(sound_file_path, aac_sound_file_path, "aac")
            sound_file_path = aac_sound_file_path

        temp_file_path = movie_file_path.with_name(
            f"{movie_file_path.stem}_temp{movie_file_path.suffix}"
        )
        # Use ffmpeg to mux video + audio
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(movie_file_path),
            "-i", str(sound_file_path),
            "-c:v", "copy",
            "-shortest",
            str(temp_file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            _sfw_logger.error(f"ffmpeg audio mux stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg audio mux failed: {result.stderr}")

        shutil.move(str(temp_file_path), str(movie_file_path))
        sound_file_path.unlink()

    self.print_file_ready_message(str(movie_file_path))
    if _sfw.write_to_movie():
        for file_path in partial_movie_files:
            _sfw.modify_atime(file_path)


# Apply patches
_sfw.SceneFileWriter.combine_files = _patched_combine_files
_sfw.SceneFileWriter.combine_to_movie = _patched_combine_to_movie

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# THEME SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Immutable color palette."""

    def __init__(self, *, name, background, text_primary, text_secondary,
                 text_muted, accent, accent2, accent3, surface, border, highlight):
        self.name = name
        self.background = background
        self.text_primary = text_primary
        self.text_secondary = text_secondary
        self.text_muted = text_muted
        self.accent = accent
        self.accent2 = accent2
        self.accent3 = accent3
        self.surface = surface
        self.border = border
        self.highlight = highlight

    def __repr__(self):
        return f"Theme({self.name})"


DARK = Theme(
    name="Dark",
    background=ManimColor("#0a0a12"),
    text_primary=ManimColor("#f0f0f0"),
    text_secondary=ManimColor("#a0a0b0"),
    text_muted=ManimColor("#606070"),
    accent=ManimColor("#4488cc"),
    accent2=ManimColor("#00ff88"),
    accent3=ManimColor("#ff5555"),
    surface=ManimColor("#14141e"),
    border=ManimColor("#2a2a3a"),
    highlight=ManimColor("#00ccff"),
)

WHITE = Theme(
    name="White",
    background=ManimColor("#f8f9fa"),
    text_primary=ManimColor("#1a1a2e"),
    text_secondary=ManimColor("#444455"),
    text_muted=ManimColor("#888899"),
    accent=ManimColor("#2266aa"),
    accent2=ManimColor("#008855"),
    accent3=ManimColor("#cc3333"),
    surface=ManimColor("#ffffff"),
    border=ManimColor("#d0d0dd"),
    highlight=ManimColor("#0099cc"),
)

CURRENT_THEME = DARK if os.environ.get("MANIM_THEME") == "dark" else WHITE


# ═══════════════════════════════════════════════════════════════════════════════
# SOC PRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

TITLE = "Hybridní model pro detekci pneumonie"
AUTHOR = "Michal Forgó"
GARANT = "Bc. Jan Boháč"
CATEGORY = "1. Matematika a data science"
REGION = "Plzeňský kraj"
YEAR = "2026"


class SocPresentation(Slide):
    """Main SOC presentation — pure 2D, light mode."""

    def construct(self):
        self.camera.background_color = CURRENT_THEME.background

        self._build_title_slide()
        self.next_slide()
        self.play(*[FadeOut(m, shift=DOWN*0.2) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)

        self._build_moores_law_slide()
        self.next_slide()
        self.play(*[FadeOut(m, shift=DOWN*0.2) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)

        self._build_block3()
        self._build_block4()
        self._build_block5()

    # ── Title slide ─────────────────────────────────────────────

    def _build_title_slide(self):
        bar = Rectangle(
            width=config.frame_width, height=0.06,
            color=CURRENT_THEME.accent, fill_color=CURRENT_THEME.accent,
            fill_opacity=1,
        )
        bar.to_edge(UP, buff=0)
        self.add(bar)

        category_text = Text(
            CATEGORY, font_size=18, color=CURRENT_THEME.accent, weight=BOLD,
        )
        category_text.to_corner(UL, buff=0.5).shift(DOWN * 0.15)
        self.add(category_text)

        title_line1 = Text(
            "Hybridní model", font_size=48,
            color=CURRENT_THEME.text_primary, weight=BOLD,
        )
        title_line2 = Text(
            "pro detekci pneumonie", font_size=48,
            color=CURRENT_THEME.accent, weight=BOLD,
        )
        title_group = VGroup(title_line1, title_line2)
        title_group.arrange(DOWN, buff=0.1, center=True)

        subtitle = Text(
            "Hybridní kvantově‑klasický model s ConvNeXt‑Tiny a VQC",
            font_size=22, color=CURRENT_THEME.text_secondary,
        )

        sep = Line(LEFT * 2.5, RIGHT * 2.5, color=CURRENT_THEME.accent, stroke_width=2)

        author_label = Text(AUTHOR, font_size=30, color=CURRENT_THEME.text_primary, weight=BOLD)
        author_role = Text("Autor", font_size=16, color=CURRENT_THEME.text_muted)
        author_col = VGroup(author_label, author_role)
        author_col.arrange(DOWN, buff=0.05, center=True)

        garant_label = Text(GARANT, font_size=24, color=CURRENT_THEME.text_secondary)
        garant_role = Text("Odborný garant", font_size=14, color=CURRENT_THEME.text_muted)
        garant_col = VGroup(garant_label, garant_role)
        garant_col.arrange(DOWN, buff=0.05, center=True)

        central = VGroup(title_group, subtitle, sep, author_col, garant_col)
        central.arrange(DOWN, buff=0.5, center=True)
        central.move_to(ORIGIN).shift(UP * 0.3)

        region_group = VGroup(
            Text(REGION, font_size=20, color=CURRENT_THEME.text_secondary),
            Text(YEAR, font_size=18, color=CURRENT_THEME.text_muted),
        )
        region_group.arrange(DOWN, center=True, buff=0.15)
        region_group.to_corner(DR, buff=0.8)

        self.play(Write(title_group, run_time=1.2))
        self.play(FadeIn(subtitle, shift=UP * 0.1, run_time=0.6), GrowFromCenter(sep, run_time=0.4))
        self.play(Write(author_col, run_time=0.7))
        self.play(Write(garant_col, run_time=0.6))
        self.play(FadeIn(region_group, shift=UP * 0.2, run_time=0.6))
        self.wait(0.5)

    # ── Moore's Law slide ───────────────────────────────────────

    def _build_moores_law_slide(self):
        header = Text(
            "Konec Moorova zákona", font_size=34,
            color=CURRENT_THEME.text_primary, weight=BOLD,
        ).to_corner(UL)
        self.play(Write(header), run_time=0.5)

        subtitle = Text(
            "a motivace pro kvantové počítání", font_size=24,
            color=CURRENT_THEME.text_secondary,
        ).next_to(header, DOWN, buff=0.15).align_to(header, LEFT)
        self.play(FadeIn(subtitle, shift=UP * 0.1), run_time=0.4)
        self.wait(0.2)

        moore_def = VGroup(
            Text("Gordon Moore (1965):", font_size=22, color=CURRENT_THEME.accent, weight=BOLD),
            Text(
                "Počet tranzistorů na čipu se zdvojnásobuje ~ každé 2 roky",
                font_size=22, color=CURRENT_THEME.text_primary,
            ),
        )
        moore_def.arrange(DOWN, buff=0.1, center=False, aligned_edge=LEFT)
        moore_def.next_to(subtitle, DOWN, buff=0.5).align_to(subtitle, LEFT)
        self.play(Write(moore_def), run_time=0.6)

        slowdown = VGroup(
            Text(
                "✔  Platil několik desetiletí, nyní zpomaluje",
                font_size=21, color=CURRENT_THEME.text_secondary,
            ),
            Text(
                "✔  Fyzikální limity křemíku: kvantové tunelování, tepelná disipace",
                font_size=21, color=CURRENT_THEME.text_secondary,
            ),
        )
        slowdown.arrange(DOWN, buff=0.1, center=False, aligned_edge=LEFT)
        slowdown.next_to(moore_def, DOWN, buff=0.3).align_to(moore_def, LEFT)
        self.play(Write(slowdown), run_time=0.6)
        self.wait(0.3)

        cons_box = Rectangle(
            width=7.5, height=1.0,
            color=CURRENT_THEME.accent2, fill_color=CURRENT_THEME.accent2,
            fill_opacity=0.08, stroke_width=2,
        )
        cons_text = Text(
            "Hledají se nová výpočetní paradigmata",
            font_size=24, color=CURRENT_THEME.accent2, weight=BOLD,
        )
        cons_group = VGroup(cons_box, cons_text)
        cons_text.move_to(cons_box.get_center())
        cons_group.next_to(slowdown, DOWN, buff=0.5).align_to(slowdown, LEFT)
        self.play(FadeIn(cons_box, scale=0.95, run_time=0.5), Write(cons_text, run_time=0.4))
        self.wait(0.3)

        q_header = Text(
            "Kvantové počítání jako alternativa", font_size=24,
            color=CURRENT_THEME.accent3, weight=BOLD,
        ).next_to(cons_group, DOWN, buff=0.5).align_to(cons_group, LEFT)
        self.play(Write(q_header), run_time=0.5)

        q_points = VGroup(
            Text("✔  Využití superpozice a entanglementu", font_size=21, color=CURRENT_THEME.text_primary),
            Text("✔  Exponenciální zrychlení pro určité třídy problémů", font_size=21, color=CURRENT_THEME.text_primary),
            Text("✔  Tento projekt: hybridní kvantově-klasický model (VQC) pro medicínu", font_size=21, color=CURRENT_THEME.accent),
        )
        q_points.arrange(DOWN, buff=0.1, center=False, aligned_edge=LEFT)
        q_points.next_to(q_header, DOWN, buff=0.2).align_to(q_header, LEFT)
        self.play(Write(q_points), run_time=0.8)

        nisq_note = Text(
            "— i v době NISQ (Noisy Intermediate-Scale Quantum)",
            font_size=18, color=CURRENT_THEME.text_muted,
        ).next_to(q_points, DOWN, buff=0.1).align_to(q_points, LEFT).shift(RIGHT * 0.3)
        self.play(FadeIn(nisq_note, shift=UP * 0.1), run_time=0.4)
        self.wait(0.5)

    # ── Block 3: Architecture + VQC ─────────────────────────────

    def _build_block3(self):
        header = Text(
            "Architektura hybridního modelu", font_size=34,
            color=CURRENT_THEME.text_primary,
        ).to_corner(UL)
        self.play(Write(header), run_time=0.5)
        self.wait(0.2)

        box_style = dict(width=1.7, height=0.9, stroke_width=2, fill_opacity=0.06)
        labels_text = [
            ("CT\nsnímek", CURRENT_THEME.text_secondary),
            ("ConvNeXt\n768 dim", CURRENT_THEME.accent),
            ("Autoenkodér\n64 dim", CURRENT_THEME.accent),
            ("VQC\n6 qubitů", CURRENT_THEME.accent3),
            ("Diagnóza\nP / N", CURRENT_THEME.accent2),
        ]
        boxes = VGroup()
        for txt, clr in labels_text:
            box = Rectangle(**box_style, color=clr, fill_color=clr)
            label = Text(txt, font_size=18, color=clr, weight=BOLD)
            group = VGroup(box, label)
            label.move_to(box.get_center())
            boxes.add(group)

        boxes.arrange(RIGHT, buff=0.4, center=True)
        boxes.move_to(ORIGIN).shift(UP * 0.3)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                start=boxes[i].get_right() + RIGHT * 0.05,
                end=boxes[i + 1].get_left() - RIGHT * 0.05,
                color=CURRENT_THEME.text_muted, stroke_width=2.5, buff=0,
            )
            arrows.add(a)

        dann_box = Rectangle(
            width=2.2, height=0.7, color=CURRENT_THEME.highlight,
            fill_color=CURRENT_THEME.highlight, fill_opacity=0.08, stroke_width=2,
        )
        dann_label = Text("DANN (domain adaptation)", font_size=16, color=CURRENT_THEME.highlight, weight=BOLD)
        dann_group = VGroup(dann_box, dann_label)
        dann_label.move_to(dann_box.get_center())
        dann_group.next_to(boxes[2], DOWN, buff=0.5)
        dann_arrow = Arrow(
            start=dann_group.get_top() + UP * 0.05,
            end=boxes[2].get_bottom() - DOWN * 0.05,
            color=CURRENT_THEME.highlight, stroke_width=2, buff=0,
        )

        self.play(FadeIn(boxes[0], scale=0.8), run_time=0.5)
        for i in range(4):
            self.play(GrowArrow(arrows[i], run_time=0.3), FadeIn(boxes[i + 1], scale=0.8, run_time=0.4))
        self.wait(0.3)
        self.play(FadeIn(dann_group, scale=0.8, run_time=0.5), GrowArrow(dann_arrow, run_time=0.3))
        self.wait(0.3)
        self.next_slide()

        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in [header, boxes, arrows, dann_group, dann_arrow]], run_time=0.5)
        self.wait(0.2)

        header2 = Text(
            "Kvantový klasifikátor a Fourierova řada", font_size=34,
            color=CURRENT_THEME.text_primary,
        ).to_corner(UL)
        self.play(Write(header2), run_time=0.5)

        layers = VGroup()
        layer_colors = [CURRENT_THEME.accent, CURRENT_THEME.highlight, CURRENT_THEME.accent3]
        for l in range(3):
            rect = Rectangle(
                width=2.0, height=1.8, color=layer_colors[l],
                fill_color=layer_colors[l], fill_opacity=0.05, stroke_width=2,
            )
            t1 = Text(f"Layer {l + 1}", font_size=16, color=layer_colors[l], weight=BOLD)
            t2 = Text("AE(x) + Rot(θ)\n+ CNOT ring", font_size=13, color=CURRENT_THEME.text_secondary)
            group = VGroup(rect, t1, t2)
            t1.next_to(rect.get_top(), DOWN, buff=0.15)
            t2.move_to(rect.get_center())
            layers.add(group)

        layers.arrange(RIGHT, buff=0.35, center=True)
        layers.move_to(ORIGIN).shift(UP * 0.4)

        meas_arrow = Arrow(
            start=layers[-1].get_right() + RIGHT * 0.2,
            end=layers[-1].get_right() + RIGHT * 1.2,
            color=CURRENT_THEME.text_primary, stroke_width=2.5,
        )
        meas_label = MathTex(r"\langle Z_0 \rangle", font_size=30, color=CURRENT_THEME.accent2)
        meas_label.next_to(meas_arrow, DOWN, buff=0.15)

        fourier_eq = MathTex(
            r"f(\mathbf{x}) = \sum_{k} c_k(\boldsymbol{\theta})"
            r" \; e^{\,i\,k\,(\mathbf{w} \cdot \mathbf{x})}",
            font_size=32, color=CURRENT_THEME.text_primary,
        )
        fourier_eq.next_to(layers, DOWN, buff=0.7)

        wi_annotation = Text(
            "wᵢ — naučitelné frekvence → analýza Fourierovy řady",
            font_size=20, color=CURRENT_THEME.accent3,
        )
        wi_annotation.next_to(fourier_eq, DOWN, buff=0.25)

        param_text = Text(
            "Parametry VQC: 62 (vs. 2 113 u MLP)",
            font_size=22, color=CURRENT_THEME.accent2, weight=BOLD,
        )
        param_text.next_to(wi_annotation, DOWN, buff=0.3)

        self.play(*[FadeIn(l, scale=0.85, run_time=0.4) for l in layers])
        self.wait(0.2)
        self.play(GrowArrow(meas_arrow, run_time=0.3), Write(meas_label, run_time=0.4))
        self.wait(0.2)
        self.play(Write(fourier_eq, run_time=0.8))
        self.wait(0.2)
        self.play(Write(wi_annotation, run_time=0.5))
        self.wait(0.15)
        self.play(Write(param_text, run_time=0.5))
        self.wait(0.5)
        self.next_slide()

        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in self.mobjects], run_time=0.5)
        self.wait(0.2)

    # ── Block 4: Results ────────────────────────────────────────

    def _build_block4(self):
        header = Text(
            "Srovnání modelů — výsledky", font_size=34,
            color=CURRENT_THEME.text_primary,
        ).to_corner(UL)
        self.play(Write(header), run_time=0.5)
        self.wait(0.2)

        col_data = [
            ("Metrika", CURRENT_THEME.text_primary, True),
            ("MLP", CURRENT_THEME.accent, True),
            ("VQC", CURRENT_THEME.accent3, True),
            ("Rozdíl", CURRENT_THEME.text_secondary, True),
        ]
        rows = [
            ("Přesnost (Accuracy)", "82.53 %", "81.25 %", "−1.28 %"),
            ("Senzitivita (Recall)", "99.74 %", "99.74 %", "0.00 %"),
            ("Specificita", "53.85 %", "50.43 %", "−3.42 %"),
            ("F1 skóre", "0.877", "0.869", "−0.008"),
            ("Parametry", "2 113", "62", "−2 051"),
        ]
        col_widths = [4.2, 1.8, 1.8, 1.8]
        row_height = 0.55

        def make_cell(text, color, bold=False, font_size=19):
            return Text(text, font_size=font_size, color=color, weight=BOLD if bold else NORMAL)

        table_group = VGroup()
        x_start = -config.frame_width / 2 + 0.8
        y_start = 2.2

        header_bg = Rectangle(
            width=sum(col_widths) + 0.4, height=row_height + 0.1,
            color=CURRENT_THEME.surface, fill_color=CURRENT_THEME.accent,
            fill_opacity=0.1, stroke_width=1.5, stroke_color=CURRENT_THEME.accent,
        )
        header_bg.move_to([x_start + sum(col_widths) / 2 + 0.2, y_start, 0])
        table_group.add(header_bg)

        cx = x_start
        for j, (val, clr, bold) in enumerate(col_data):
            cell = make_cell(val, clr, bold, font_size=18)
            cell.move_to([cx + col_widths[j] / 2, y_start, 0])
            table_group.add(cell)
            cx += col_widths[j] + 0.1

        for i, row in enumerate(rows):
            yy = y_start - (i + 1) * row_height - 0.15
            if i % 2 == 0:
                bg = Rectangle(
                    width=sum(col_widths) + 0.4, height=row_height,
                    color=CURRENT_THEME.surface, fill_color=CURRENT_THEME.text_muted,
                    fill_opacity=0.04, stroke_width=0,
                )
                bg.move_to([x_start + sum(col_widths) / 2 + 0.2, yy, 0])
                table_group.add(bg)

            metric_label = Text(row[0], font_size=17, color=CURRENT_THEME.text_primary)
            metric_label.move_to([x_start + 0.1, yy, 0], aligned_edge=LEFT)
            metric_label.shift(RIGHT * 0.3)
            table_group.add(metric_label)

            for j, val in enumerate(row[1:], start=1):
                clr = CURRENT_THEME.accent if j == 1 else CURRENT_THEME.accent3 if j == 2 else CURRENT_THEME.text_primary
                cell = make_cell(val, clr, font_size=17)
                cell.move_to([x_start + sum(col_widths[:j]) + col_widths[j] / 2 + 0.1 * j, yy, 0])
                table_group.add(cell)

        self.play(FadeIn(table_group, shift=UP * 0.2, run_time=0.8))
        self.wait(0.5)

        eff_box = Rectangle(
            width=6.0, height=0.6, color=CURRENT_THEME.accent2,
            fill_color=CURRENT_THEME.accent2, fill_opacity=0.1, stroke_width=2,
        )
        eff_text = Text(
            "34× méně parametrů — rozdíl pouze 1.28 %",
            font_size=22, color=CURRENT_THEME.accent2, weight=BOLD,
        )
        eff_group = VGroup(eff_box, eff_text)
        eff_text.move_to(eff_box.get_center())
        eff_group.next_to(table_group, DOWN, buff=0.4)
        self.play(FadeIn(eff_group, scale=0.9, run_time=0.6))
        self.wait(0.5)
        self.next_slide()

        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in [header, table_group, eff_group]], run_time=0.4)
        self.wait(0.2)

        header2 = Text(
            "Statistická signifikance", font_size=34,
            color=CURRENT_THEME.text_primary,
        ).to_corner(UL)
        self.play(Write(header2), run_time=0.5)

        mcnemar_title = Text(
            "McNemarův test (α = 0.05)", font_size=26,
            color=CURRENT_THEME.accent, weight=BOLD,
        ).move_to(UP * 1.5)
        self.play(Write(mcnemar_title), run_time=0.4)

        mcnemar_body = VGroup(
            MathTex(r"\chi^2 = 0.672,\quad p = 0.412", font_size=30, color=CURRENT_THEME.text_primary),
            Text("p > α  →  rozdíl není statisticky významný", font_size=22, color=CURRENT_THEME.accent2),
        )
        mcnemar_body.arrange(DOWN, buff=0.2, center=True)
        mcnemar_body.next_to(mcnemar_title, DOWN, buff=0.3)
        self.play(Write(mcnemar_body), run_time=0.6)
        self.wait(0.3)

        ci_title_boot = Text(
            "Bootstrap 95% IS (B = 1 000)", font_size=26,
            color=CURRENT_THEME.accent, weight=BOLD,
        ).next_to(mcnemar_body, DOWN, buff=0.6)
        self.play(Write(ci_title_boot), run_time=0.4)

        ci_body = VGroup(
            MathTex(r"\text{MLP: } [0.918,\; 0.957]", font_size=28, color=CURRENT_THEME.accent),
            MathTex(r"\text{VQC: } [0.827,\; 0.889]", font_size=28, color=CURRENT_THEME.accent3),
            Text("Intervaly se překrývají → potvrzení srovnatelnosti", font_size=20, color=CURRENT_THEME.text_secondary),
        )
        ci_body.arrange(DOWN, buff=0.15, center=True)
        ci_body.next_to(ci_title_boot, DOWN, buff=0.3)
        self.play(Write(ci_body), run_time=0.7)
        self.wait(0.5)
        self.next_slide()

        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in self.mobjects], run_time=0.5)
        self.wait(0.2)

    # ── Block 5: Conclusion ─────────────────────────────────────

    def _build_block5(self):
        header = Text(
            "Interpretovatelnost — Grad‑CAM", font_size=34,
            color=CURRENT_THEME.text_primary,
        ).to_corner(UL)
        self.play(Write(header), run_time=0.5)

        xray_placeholder = Rectangle(
            width=4.0, height=4.0, color=CURRENT_THEME.border,
            fill_color=CURRENT_THEME.surface, fill_opacity=0.5, stroke_width=2,
        )
        xray_placeholder.shift(LEFT * 2.5 + DOWN * 0.2)
        xray_label = Text("RTG snímek + heatmapa\n(Grad‑CAM)", font_size=18, color=CURRENT_THEME.text_secondary)
        xray_label.move_to(xray_placeholder.get_center())
        xray_group = VGroup(xray_placeholder, xray_label)
        self.play(FadeIn(xray_group, scale=0.9, run_time=0.6))

        explanation = VGroup(
            Text("Zaměření na plicní parenchym", font_size=24, color=CURRENT_THEME.text_primary, weight=BOLD),
            Text("Oblasti konsolidace → bakteriální pneumonie", font_size=20, color=CURRENT_THEME.text_secondary),
            Text("Konvoluční vrstvy → prostorová mapa aktivace", font_size=18, color=CURRENT_THEME.text_muted),
        )
        explanation.arrange(DOWN, buff=0.2, center=True)
        explanation.next_to(xray_group, RIGHT, buff=0.8)
        explanation.shift(UP * 0.2)
        self.play(Write(explanation, run_time=0.8))
        self.wait(0.5)
        self.next_slide()

        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in [header, xray_group, explanation]], run_time=0.4)
        self.wait(0.2)

        header2 = Text(
            "Připraveno pro IBM Quantum", font_size=34,
            color=CURRENT_THEME.text_primary,
        ).to_corner(UL)
        self.play(Write(header2), run_time=0.5)

        sim_box = Rectangle(
            width=2.4, height=1.2, color=CURRENT_THEME.accent,
            fill_color=CURRENT_THEME.accent, fill_opacity=0.06, stroke_width=2,
        )
        sim_text = Text("Ideální\nsimulátor", font_size=20, color=CURRENT_THEME.accent, weight=BOLD)
        sim_text.move_to(sim_box.get_center())
        sim = VGroup(sim_box, sim_text)

        ibm_box = Rectangle(
            width=2.4, height=1.2, color=CURRENT_THEME.accent3,
            fill_color=CURRENT_THEME.accent3, fill_opacity=0.06, stroke_width=2,
        )
        ibm_text = Text("IBM Quantum\nHeron r2 (156 q)", font_size=20, color=CURRENT_THEME.accent3, weight=BOLD)
        ibm_text.move_to(ibm_box.get_center())
        ibm = VGroup(ibm_box, ibm_text)

        zne_box = Rectangle(
            width=2.4, height=1.2, color=CURRENT_THEME.accent2,
            fill_color=CURRENT_THEME.accent2, fill_opacity=0.06, stroke_width=2,
        )
        zne_text = Text("ZNE\nextrapolace", font_size=20, color=CURRENT_THEME.accent2, weight=BOLD)
        zne_text.move_to(zne_box.get_center())
        zne = VGroup(zne_box, zne_text)

        pipeline = VGroup(sim, ibm, zne)
        pipeline.arrange(RIGHT, buff=0.6, center=True)
        pipeline.move_to(ORIGIN).shift(UP * 0.3)

        arr1 = Arrow(sim.get_right(), ibm.get_left(), color=CURRENT_THEME.text_muted, stroke_width=2.5)
        arr2 = Arrow(ibm.get_right(), zne.get_left(), color=CURRENT_THEME.text_muted, stroke_width=2.5)

        self.play(FadeIn(sim, scale=0.85, run_time=0.5))
        self.play(GrowArrow(arr1, run_time=0.3), FadeIn(ibm, scale=0.85, run_time=0.5))
        self.play(GrowArrow(arr2, run_time=0.3), FadeIn(zne, scale=0.85, run_time=0.5))

        zne_formula = MathTex(
            r"\langle O \rangle^* = \lim_{\lambda \to 0} \langle O \rangle_\lambda",
            font_size=28, color=CURRENT_THEME.text_primary,
        )
        zne_formula.next_to(zne, DOWN, buff=0.5)
        zne_note = Text(
            "Zero‑Noise Extrapolation: gate‑folding {1×, 2×, 3×}",
            font_size=18, color=CURRENT_THEME.text_secondary,
        )
        zne_note.next_to(zne_formula, DOWN, buff=0.2)

        self.play(Write(zne_formula, run_time=0.6), FadeIn(zne_note, shift=UP * 0.1, run_time=0.5))
        self.wait(0.5)
        self.next_slide()

        self.play(*[FadeOut(m, shift=DOWN * 0.15) for m in [header2, sim, ibm, zne, arr1, arr2, zne_formula, zne_note]], run_time=0.4)
        self.wait(0.2)

        closing_title = Text("Shrnutí", font_size=42, color=CURRENT_THEME.text_primary, weight=BOLD)
        closing_title.to_edge(UP, buff=1.0)
        self.play(Write(closing_title), run_time=0.6)

        bullets = [
            "Srovnatelná přesnost: rozdíl pouze 1.28 %",
            "34× méně parametrů (62 vs. 2 113)",
            "Statisticky nesignifikantní (McNemar: p = 0.412)",
            "Připraveno pro IBM Quantum + ZNE mitigaci",
            "Grad‑CAM interpretace pro klinickou praxi",
        ]
        bullet_group = VGroup()
        for b in bullets:
            line = VGroup(
                Text("▸ ", font_size=24, color=CURRENT_THEME.accent),
                Text(b, font_size=24, color=CURRENT_THEME.text_primary),
            )
            line.arrange(RIGHT, buff=0.1, center=False)
            bullet_group.add(line)
        bullet_group.arrange(DOWN, buff=0.25, center=False, aligned_edge=LEFT)
        bullet_group.move_to(ORIGIN).shift(DOWN * 0.2)

        self.play(Write(bullet_group, run_time=1.2))
        self.wait(0.5)

        thanks = Text("Děkuji za pozornost", font_size=36, color=CURRENT_THEME.accent2, weight=BOLD)
        thanks.next_to(bullet_group, DOWN, buff=0.6)
        self.play(Write(thanks, run_time=0.7))
        self.wait(0.8)
        self.next_slide()

        self.play(*[FadeOut(m, run_time=0.6) for m in self.mobjects])
        self.wait(0.2)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCH SPHERE PRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

SPHERE_COLOR    = ManimColor("#4488CC")
AXIS_COLOR      = ManimColor("#CCCCCC")
Z_AXIS_COLOR    = ManimColor("#FFD700")
STATE_0_COLOR   = ManimColor("#00FF88")
STATE_1_COLOR   = ManimColor("#FF5555")
STATE_MIX_COLOR = ManimColor("#FFAA00")


class BlochSpherePresentation(ThreeDSlide):
    """Interactive Bloch sphere slides: |0⟩ → |1⟩ transition."""

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        self.camera.set_zoom(1.1)

        title = Text("Bloch Sphere", font_size=60, color=WHITE)
        title.shift(UP * 0.5)
        subtitle = Text(
            "Geometric representation of a qubit state",
            font_size=28, color=GREY_A,
        )
        subtitle.next_to(title, DOWN, buff=0.4)
        eq = MathTex(
            r"\vert\psi\rangle = \alpha\vert 0\rangle + \beta\vert 1\rangle",
            font_size=38,
        )
        eq.next_to(subtitle, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.6)
        self.play(Write(eq), run_time=0.8)
        self.wait(0.3)
        self.next_slide()
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(eq))

        header = Text("The Bloch Sphere", font_size=40, color=WHITE).to_corner(UL)
        self.add_fixed_in_frame_mobjects(header)
        self.play(Write(header), run_time=0.5)

        sphere = self._build_sphere()
        axes, xl, yl, zl = self._build_axes()
        self.play(
            Create(sphere, run_time=1.2),
            *[Create(a) for a in axes if isinstance(a, Arrow3D)],
            FadeIn(xl), FadeIn(yl), FadeIn(zl),
        )

        vec = self._build_arrow(np.array([0, 0, 1]), STATE_0_COLOR)
        vlabel = MathTex(r"\vert 0\rangle", font_size=38, color=STATE_0_COLOR)
        vlabel.next_to(np.array([0, 0, 2.3]), LEFT, buff=0.2)
        self.play(Create(vec), Write(vlabel), run_time=0.8)
        self.wait(0.3)
        self.next_slide()

        state_eq = MathTex(
            r"\vert\psi\rangle = \vert 0\rangle",
            font_size=36, color=STATE_0_COLOR,
        ).to_corner(DR)
        self.add_fixed_in_frame_mobjects(state_eq)
        self.play(Write(state_eq), run_time=0.5)
        self.wait(0.3)
        self.next_slide()

        self.move_camera(phi=60 * DEGREES, theta=-135 * DEGREES, run_time=1.0)
        self.next_slide()

        t_tracker = ValueTracker(0.0)
        actual_arrow = self._build_arrow(np.array([0, 0, 1]), STATE_0_COLOR)
        actual_arrow.add_updater(
            lambda mobj, dt: mobj.become(self._build_arrow(
                self._state_at(t_tracker.get_value()),
                self._interp_color(t_tracker.get_value()),
            ))
        )
        self.add(actual_arrow)
        self.remove(vec)
        self.wait(0.1)

        self.play(t_tracker.animate.set_value(1.0), run_time=3.0, rate_func=linear)
        actual_arrow.clear_updaters()
        self.wait(0.2)

        new_state_eq = MathTex(
            r"\vert\psi\rangle = \vert 1\rangle",
            font_size=36, color=STATE_1_COLOR,
        ).to_corner(DR)
        self.play(Transform(state_eq, new_state_eq), FadeOut(vlabel), run_time=0.5)

        final_label = MathTex(r"\vert 1\rangle", font_size=42, color=STATE_1_COLOR)
        final_label.next_to(np.array([0, 0, -2.3]), LEFT, buff=0.2)
        self.play(Write(final_label), run_time=0.5)
        self.wait(0.3)
        self.next_slide()

        self.move_camera(phi=70 * DEGREES, theta=-45 * DEGREES, run_time=1.0)

        summary = VGroup(
            Text(r"|0⟩  →  north pole  →  ground state", font_size=28, color=STATE_0_COLOR),
            Text(r"|1⟩  →  south pole  →  excited state", font_size=28, color=STATE_1_COLOR),
            MathTex(
                r"\vert\psi\rangle = \cos\frac{\theta}{2}\vert 0\rangle"
                r" + e^{i\phi}\sin\frac{\theta}{2}\vert 1\rangle",
                font_size=30,
            ),
        )
        summary.arrange(DOWN, center=True, buff=0.3)
        summary.to_corner(UL, buff=0.4)
        self.add_fixed_in_frame_mobjects(summary)
        self.play(Write(summary), run_time=1.0)
        self.wait(0.5)
        self.next_slide()

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.wait(0.2)

    def _build_sphere(self):
        s = Sphere(radius=2.0, resolution=(12, 8))
        s.set_fill(opacity=0.06)
        s.set_stroke(color=SPHERE_COLOR, opacity=0.35, width=1.0)
        return s

    def _build_axes(self):
        axes = VGroup()
        for start, end, color, thick in [
            ([0, 0, -2.4], [0, 0, 2.4], Z_AXIS_COLOR, 0.04),
            ([-2.4, 0, 0], [2.4, 0, 0], AXIS_COLOR, 0.025),
            ([0, -2.4, 0], [0, 2.4, 0], AXIS_COLOR, 0.025),
        ]:
            a = Arrow3D(start=np.array(start), end=np.array(end), color=color, thickness=thick)
            axes.add(a)
        xl = MathTex("x", font_size=26, color=AXIS_COLOR).move_to([2.7, 0, 0])
        yl = MathTex("y", font_size=26, color=AXIS_COLOR).move_to([0, 2.7, 0])
        zl = MathTex("z", font_size=26, color=Z_AXIS_COLOR).move_to([0, 0, 2.7])
        return axes, xl, yl, zl

    def _build_arrow(self, direction, color):
        d = direction / np.linalg.norm(direction)
        return Arrow3D(start=np.array([0, 0, 0]), end=d * 2.0, color=color, thickness=0.06)

    @staticmethod
    def _state_at(t):
        theta = t * PI
        return np.array([np.sin(theta), 0.0, np.cos(theta)])

    @staticmethod
    def _interp_color(t):
        if t <= 0.5:
            return interpolate_color(STATE_0_COLOR, STATE_MIX_COLOR, t * 2)
        return interpolate_color(STATE_MIX_COLOR, STATE_1_COLOR, (t - 0.5) * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — render, present, convert, merge
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = sys.executable


def manim_slides_bin():
    """Locate manim-slides binary."""
    venv_dir = os.path.dirname(os.path.dirname(VENV_PYTHON))
    cand = os.path.join(venv_dir, "bin", "manim-slides")
    if os.path.exists(cand):
        return cand
    return "manim-slides"


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def cmd_render(class_name, quality="l"):
    """Render a presentation class with manim-slides."""
    result = run(
        [manim_slides_bin(), "render", f"-q{quality}", "presentation.py", class_name],
        cwd=SCRIPT_DIR,
    )
    return result.returncode == 0


def cmd_present(class_name):
    """Present slides interactively."""
    return run(
        [manim_slides_bin(), "present", "presentation.py", class_name],
        cwd=SCRIPT_DIR,
    )


def cmd_html(class_name, out_file):
    """Convert rendered slides to self-contained HTML."""
    result = run(
        [manim_slides_bin(), "convert", "--to", "html", "--one-file", class_name, out_file],
        cwd=SCRIPT_DIR,
    )
    return result.returncode == 0


def cmd_merge_toggle(dark_html_path, white_html_path, out_path):
    """Merge dark + white HTML into a single self-contained toggle file."""
    with open(dark_html_path, encoding="utf-8") as f:
        dark_html = f.read()
    with open(white_html_path, encoding="utf-8") as f:
        white_html = f.read()

    def extract_sections(html):
        marker = '<div class="slides">'
        start = html.index(marker) + len(marker)
        slides_area = html[start:]
        results = []
        for part in slides_area.split("</section>"):
            if "<section" not in part:
                continue
            sec_start = part.index("<section")
            sec_html = part[sec_start:] + "</section>"
            bg = re.search(r'data-background-color="([^"]+)"', sec_html)
            vd = re.search(r'data-background-video="([^"]+)"', sec_html)
            results.append((
                bg.group(1) if bg else "#000000",
                vd.group(1) if vd else "",
            ))
        return results

    dark_slides = extract_sections(dark_html)
    white_slides = extract_sections(white_html)
    n = min(len(dark_slides), len(white_slides))
    dark_slides = dark_slides[:n]
    white_slides = white_slides[:n]

    def esc(s):
        return s.replace("\\", "\\\\").replace("'", "\\'")

    def arr(items):
        return ", ".join(f"'{esc(it)}'" for it in items)

    wb = arr([b for b, _ in white_slides])
    wv = arr([v for _, v in white_slides])
    db = arr([b for b, _ in dark_slides])
    dv = arr([v for _, v in dark_slides])

    def mk_slide(bg, vid):
        return (
            "        <section\n"
            "          data-background-size='contain'\n"
            f"          data-background-color=\"{bg}\"\n"
            f"          data-background-video=\"{vid}\"\n"
            "          data-background-video-muted\n"
            "          >\n"
            "        </section>"
        )

    slides_html = "\n".join(mk_slide(b, v) for b, v in dark_slides)

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Hybridní model pro detekci pneumonie — SOČ 2026</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.2.0/reveal.min.css">
    <link rel="stylesheet" id="theme-css" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.2.0/theme/black.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.13.1/styles/zenburn.min.css">
    <style>
      #toggle-zone {{
        position: fixed; bottom: 0; right: 0; z-index: 9999;
        padding: 40px;
      }}
      #theme-toggle {{
        opacity: 0;
        background: rgba(255,255,255,0.15); backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.25); border-radius: 40px;
        padding: 10px 18px; cursor: pointer;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 15px; font-weight: 500; color: #f0f0f0;
        display: flex; align-items: center; gap: 8px;
        transition: opacity 0.3s ease, background 0.3s ease, transform 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        user-select: none;
        pointer-events: none;
      }}
      #toggle-zone:hover #theme-toggle {{ opacity: 1; pointer-events: auto; }}
      #theme-toggle:hover {{ background: rgba(255,255,255,0.25); transform: scale(1.05); }}
      body.light-mode #theme-toggle {{
        background: rgba(0,0,0,0.08); border-color: rgba(0,0,0,0.15);
        color: #1a1a2e; box-shadow: 0 4px 16px rgba(0,0,0,0.1);
      }}
      body.light-mode #theme-toggle:hover {{ background: rgba(0,0,0,0.15); }}
    </style>
  </head>
  <body>
    <div class="reveal">
      <div class="slides" id="slides-container">
{slides_html}
      </div>
    </div>

    <div id="toggle-zone">
      <button id="theme-toggle" onclick="toggleTheme()" aria-label="Přepnout motiv">
        <span class="icon" id="toggle-icon">&#127769;</span>
        <span id="toggle-label">White</span>
      </button>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.2.0/reveal.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.2.0/plugin/notes/notes.min.js"></script>

    <script>
    // ── Both themes embedded inline — fully self-contained ──
    var WHITE_BG = [{wb}];
    var WHITE_VIDEO = [{wv}];
    var DARK_BG = [{db}];
    var DARK_VIDEO = [{dv}];
    var N = {n};
    var theme = 'dark';

    function init() {{
      Reveal.initialize({{
        width:'100%', height:'100%', margin:0.04,
        minScale:0.2, maxScale:2.0,
        controls:false, progress:false, slideNumber:false, hash:false,
        keyboard:true, touch:true, loop:false,
        transition:'none', backgroundTransition:'none',
        viewDistance:3, hideInactiveCursor:true, hideCursorTime:5000
      }});

      Reveal.addKeyBinding({{keyCode:32,key:'SPACE',description:'Play/pause'}},function(){{
        var v=Reveal.getCurrentSlide().slideBackgroundContentElement.getElementsByTagName('video');
        if(v.length>0){{if(v[0].paused)v[0].play();else v[0].pause();}}else Reveal.next();
      }});

      function fixVid(v){{
        var s=v.querySelectorAll('source');
        for(var i=0;i<s.length;i++){{
          var src=s[i].getAttribute('src');
          if(src&&src.match(/^data:video.*;base64$/)){{
            var n=s[i+1];
            if(n) v.setAttribute('src',src+','+n.getAttribute('src'));
          }}
        }}
      }}

      function fixAll(){{
        var bg=Reveal.getBackgroundsElement().querySelectorAll('.slide-background');
        for(var j=0;j<bg.length;j++){{(function(sl){{
          var v=sl.querySelector('video');
          if(v) fixVid(v);
          else {{
            var mo=new MutationObserver(function(m){{
              for(var k=0;k<m.length;k++){{
                if(m[k].type==='childList'){{
                  for(var l=0;l<m[k].addedNodes.length;l++){{
                    if(m[k].addedNodes[l].tagName==='VIDEO'){{fixVid(m[k].addedNodes[l]);mo.disconnect();}}
                  }}
                }}
              }}
            }});
            mo.observe(sl,{{childList:true,subtree:true}});
          }}
        }})(bg[j]);}}
      }}

      Reveal.on('ready',fixAll);
    }}

    function apply(t){{
      var bg=t==='dark'?DARK_BG:WHITE_BG;
      var vi=t==='dark'?DARK_VIDEO:WHITE_VIDEO;
      var css=t==='dark'?'black.min.css':'white.min.css';
      document.getElementById('theme-css').href='https://cdnjs.cloudflare.com/ajax/libs/reveal.js/5.2.0/theme/'+css;
      var secs=document.querySelectorAll('#slides-container section');
      for(var i=0;i<secs.length&&i<N;i++){{
        secs[i].setAttribute('data-background-color',bg[i]);
        secs[i].setAttribute('data-background-video',vi[i]);
      }}
      if(t==='white') document.body.classList.add('light-mode');
      else document.body.classList.remove('light-mode');
      var ic=document.getElementById('toggle-icon');
      var lb=document.getElementById('toggle-label');
      if(t==='white'){{ic.textContent='\\u2600\\ufe0f';lb.textContent='Dark';}}
      else{{ic.textContent='\\uD83C\\uDF19';lb.textContent='White';}}
    }}

    function doToggle(next){{
      var idx=Reveal.getIndices().h;
      Reveal.destroy();
      apply(next);
      init();
      Reveal.on('ready',function(){{Reveal.slide(idx);}});
      theme=next;
    }}

    function toggleTheme(){{
      doToggle(theme==='dark'?'white':'dark');
    }}

    init();
    </script>
  </body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) // 1024
    print(f"{os.path.basename(out_path)} → {size_kb} KB (self-contained, both themes)")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "render":
        cls = sys.argv[2] if len(sys.argv) > 2 else "SocPresentation"
        cmd_render(cls, "l")

    elif command == "render-m":
        cls = sys.argv[2] if len(sys.argv) > 2 else "SocPresentation"
        cmd_render(cls, "m")

    elif command == "present":
        cls = sys.argv[2] if len(sys.argv) > 2 else "SocPresentation"
        cmd_present(cls)

    elif command == "present-bloch":
        cmd_present("BlochSpherePresentation")

    elif command == "html":
        cls = sys.argv[2] if len(sys.argv) > 2 else "SocPresentation"
        out = sys.argv[3] if len(sys.argv) > 3 else f"{cls.lower()}.html"
        cmd_html(cls, out)

    elif command == "all":
        tmp_dark = os.path.join(SCRIPT_DIR, "_soc_dark.html")
        tmp_white = os.path.join(SCRIPT_DIR, "_soc_white.html")
        out_html = os.path.join(SCRIPT_DIR, "soc_toggle.html")

        print("=== Render dark mode ===")
        env = os.environ.copy()
        env["MANIM_THEME"] = "dark"
        subprocess.run([sys.executable, __file__, "render", "SocPresentation"], cwd=SCRIPT_DIR, env=env, check=True)
        subprocess.run([sys.executable, __file__, "html", "SocPresentation", "_soc_dark.html"], cwd=SCRIPT_DIR, env=env, check=True)

        print("=== Render white mode ===")
        env.pop("MANIM_THEME")
        subprocess.run([sys.executable, __file__, "render", "SocPresentation"], cwd=SCRIPT_DIR, env=env, check=True)
        subprocess.run([sys.executable, __file__, "html", "SocPresentation", "_soc_white.html"], cwd=SCRIPT_DIR, env=env, check=True)

        print("=== Merge into self-contained toggle ===")
        cmd_merge_toggle(tmp_dark, tmp_white, out_html)

        os.unlink(tmp_dark)
        os.unlink(tmp_white)
        print("Temp files cleaned")

    elif command == "all-bloch":
        print("=== Rendering BlochSpherePresentation ===")
        if cmd_render("BlochSpherePresentation", "l"):
            print("=== HTML export ===")
            cmd_html("BlochSpherePresentation", "bloch_sphere.html")
            print("Done: bloch_sphere.html")

    elif command in ("-h", "--help"):
        print(__doc__)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
