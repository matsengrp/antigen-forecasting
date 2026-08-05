"""One definition of the supplementary-figure look.

The aggregated supplement figures (S4, S5) settled on a convention that the newer
figures had started to drift from: seaborn's paper style with enlarged axis and
tick labels, no per-axes titles, and panel letters placed as figure text above
each panel's y-axis label rather than as a left-aligned axes title. The zoom-in
figures meanwhile passed font sizes explicitly as function arguments, which is a
second way of saying the same thing and drifts independently.

This module holds the numbers once so both styles agree. Figures that set style
globally call :func:`apply_supplement_style`; functions that take explicit font
sizes default to the constants below.

There is deliberately no canvas-scaling helper here. An earlier version scaled
these sizes by figure width, on the theory that a 30-inch figure needs larger
type than a 12-inch one to survive being scaled into the column. That is wrong:
the 30-inch figures are grids of *many small* panels, each about the size of one
panel on a 12-inch figure, so they want the same type size. Applying the scaling
tripled the fonts on figures 4 and S6 and made their titles, tick labels and
legends overlap.

The values match ``manuscript-figure-S5-growth-rate-benchmark-aggregated.ipynb``
exactly, so adopting this module does not change how S4 or S5 render.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Seaborn's stripplot jitters points using the global numpy RNG and exposes no
# seed of its own, so without this every render moved ~0.7% of the pixels in any
# figure carrying a strip overlay -- a figure could not be regenerated to match
# what was submitted. Seed immediately before each stripplot call.
JITTER_SEED = 5

# Font sizes, for callers that pass them explicitly rather than via rcParams.
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13
PANEL_LETTER_FONTSIZE = 18

# Gap between a panel's top edge and its letter, in points. This was a figure
# fraction (0.008), which made the gap depend on canvas height rather than on
# type size: 2.1pt on S1's 3.6in canvas against 10.4pt on figure 4's 18in one.
# The short figures were the ones whose letters collided with their tick labels.
#
# 6pt rather than 8: the figures are saved with bbox_inches="tight", so raising
# the letters also raises the saved bounding box. At 8pt the supplement figures
# grew by 3-5pt each and the manuscript gained a page; 6pt clears every tick
# label and leaves the page count unchanged.
PANEL_LETTER_PAD_PT = 6.0

# The base matplotlib style the supplement figures build on.
BASE_STYLE = "seaborn-v0_8-paper"

SUPPLEMENT_RC = {
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "legend.title_fontsize": LEGEND_TITLE_FONTSIZE,
}


def apply_supplement_style() -> None:
    """Set the supplement look globally, as the S4 and S5 notebooks do."""
    plt.style.use(BASE_STYLE)
    plt.rcParams.update(SUPPLEMENT_RC)


def seed_jitter() -> None:
    """Make the next seaborn strip/swarm plot's jitter reproducible.

    Call immediately before the plotting call. Seeding once at import would not
    work: any RNG use in between would shift the draw.
    """
    np.random.seed(JITTER_SEED)


def style_panel(ax: Axes) -> None:
    """Apply the shared per-axes treatment: light y-grid behind the data, despined.

    Panels carry no title; the figure's panel letter and the caption identify
    them, which is what keeps a multi-panel supplement figure readable.

    The grid is horizontal only. Gridlines were inconsistent across the figure set
    -- some panels had both axes, some only y, some none at all, and S5 differed
    between its own two panels. Horizontal lines earn their keep by letting a
    reader carry a value across a panel; vertical ones over a categorical axis are
    just texture, so they are off everywhere.
    """
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def hollow_boxes(ax: Axes, linewidth: float = 2.0) -> None:
    """Knock out every box's face and move its colour to the edge.

    The supplement's box-and-strip panels are drawn as a coloured box over a
    matching point cloud. Filling the box hides the points inside the IQR, which
    is where most of them are, so the fill is turned off and the palette colour
    becomes the outline instead. This is the look S5 panel A and figure 4/S6
    panel A settled on.

    Draw the boxplot with a palette (not ``color="0.85"``) and call this
    immediately afterwards. Boxes drawn as a flat grey do work here, but they
    come out uniformly grey-outlined, which defeats the point.

    Args:
        ax: The axes holding the boxplot patches.
        linewidth: Outline width for the boxes.
    """
    for patch in ax.patches:
        face = patch.get_facecolor()
        # Skip a box that has already been hollowed. Without this the second call
        # copies the cleared (fully transparent) face onto the edge and the
        # outline disappears, which is easy to trigger now that five figures
        # share this helper and a panel builder may be applied twice.
        if face[3] == 0.0:
            continue
        patch.set_edgecolor(face)
        patch.set_facecolor("none")
        patch.set_linewidth(linewidth)
    plt.setp(ax.lines, linewidth=1.5)


def add_panel_letters(
    fig: Figure, axes, letters: str, fontsize: int = PANEL_LETTER_FONTSIZE
) -> None:
    """Place panel letters above each panel, centred on its y-axis label.

    Lifted from the S4/S5 notebooks so every supplement figure positions its
    letters identically. Must be called after the layout is final (that is, after
    ``fig.tight_layout()``), because it measures the rendered label extents.

    Rows are handled independently, so a 2x2 grid gets its lower letters above the
    lower row rather than all four pinned to the top of the figure.

    Args:
        fig: The figure being annotated.
        axes: Flat or 2-D array of axes, in reading order.
        letters: One character per axes, e.g. ``"ABCD"``.
        fontsize: Panel-letter size.
    """
    flat = list(axes.flat) if hasattr(axes, "flat") else list(axes)
    add_split_panel_letters(fig, flat, flat, letters, fontsize)


def add_split_panel_letters(
    fig: Figure,
    label_axes,
    top_axes,
    letters: str,
    fontsize: int = PANEL_LETTER_FONTSIZE,
) -> None:
    """Place panel letters when the y-label and the top edge are on different axes.

    A panel with a broken y-axis is built from two stacked axes: the data sits on
    the lower one, which carries the y-label, while the upper one holds the
    off-scale reference line and owns the panel's top edge. Letters therefore need
    the horizontal anchor from one axes and the vertical anchor from another.

    :func:`add_panel_letters` is the ordinary case and simply passes the same axes
    for both.

    Args:
        fig: The figure being annotated.
        label_axes: Per letter, the axes whose y-label sets the horizontal anchor.
        top_axes: Per letter, the axes whose top edge sets the vertical anchor.
        letters: One character per panel, e.g. ``"ABCD"``.
        fontsize: Panel-letter size.
    """
    label_axes = list(label_axes)
    top_axes = list(top_axes)
    assert len(label_axes) == len(top_axes) == len(letters), (
        f"{len(label_axes)} label axes, {len(top_axes)} top axes, "
        f"{len(letters)} letters"
    )
    fig.canvas.draw()
    inverse = fig.transFigure.inverted()
    # Convert the pad from points to a figure fraction, so the gap is the same
    # physical size whatever the canvas height.
    pad = PANEL_LETTER_PAD_PT / (72.0 * fig.get_figheight())

    # Anchor each letter to its own panel rather than to the top of a row. An
    # earlier version measured the row and offset upward from there, which put
    # the lower letters of a grid into the gap already occupied by the row
    # above's tick labels; how bad the collision looked then depended on the
    # figure's height, so it kept reappearing after resizes.
    for index, (label_ax, top_ax) in enumerate(zip(label_axes, top_axes)):
        label_extent = label_ax.yaxis.get_label().get_window_extent()
        # Horizontally: centred on the y-axis label, matching S4 and S5.
        x_fig, _ = inverse.transform((0.5 * (label_extent.x0 + label_extent.x1), 0))
        # Vertically: pinned to this panel's own top edge, so the spacing between
        # rows is the only thing that has to be large enough.
        _, y_fig = inverse.transform((0, top_ax.get_window_extent().y1))
        fig.text(
            x_fig,
            y_fig + pad,
            letters[index],
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
