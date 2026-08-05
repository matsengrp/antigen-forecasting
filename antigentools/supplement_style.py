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

The values match ``manuscript-figure-S5-growth-rate-benchmark-aggregated.ipynb``
exactly, so adopting this module does not change how S4 or S5 render.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Font sizes, for callers that pass them explicitly rather than via rcParams.
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 13
PANEL_LETTER_FONTSIZE = 18

# The base matplotlib style the supplement figures build on.
BASE_STYLE = "seaborn-v0_8-paper"

SUPPLEMENT_RC = {
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "legend.title_fontsize": LEGEND_TITLE_FONTSIZE,
}


# Canvas width the font sizes above are calibrated for, in inches. Figures drawn
# much wider than this are scaled down more aggressively when placed in the
# document, so a literal copy of the sizes would render proportionally smaller.
REFERENCE_CANVAS_WIDTH_IN = 12.0


def scale_for_canvas(width_inches: float) -> dict[str, int]:
    """Return the supplement font sizes rescaled for a canvas of this width.

    Font sizes are only meaningful relative to the canvas they sit on, because
    the document scales every figure to the column. The zoom-in figures are drawn
    30 inches wide against roughly 12 for the aggregated panels, so copying the
    raw sizes across would leave their labels reaching the page at about 3 points
    where the others reach 8. Scaling by canvas width keeps the *rendered* size
    consistent, which is the thing a reader actually sees.

    Args:
        width_inches: Width of the figure this styling is for.

    Returns:
        Keys ``label``, ``tick``, ``legend``, ``legend_title``, ``panel_letter``.
    """
    assert width_inches > 0, f"canvas width must be positive, got {width_inches}"
    factor = width_inches / REFERENCE_CANVAS_WIDTH_IN
    return {
        "label": round(LABEL_FONTSIZE * factor),
        "tick": round(TICK_FONTSIZE * factor),
        "legend": round(LEGEND_FONTSIZE * factor),
        "legend_title": round(LEGEND_TITLE_FONTSIZE * factor),
        "panel_letter": round(PANEL_LETTER_FONTSIZE * factor),
    }


def apply_supplement_style() -> None:
    """Set the supplement look globally, as the S4 and S5 notebooks do."""
    plt.style.use(BASE_STYLE)
    plt.rcParams.update(SUPPLEMENT_RC)


def style_panel(ax: Axes) -> None:
    """Apply the shared per-axes treatment: light grid behind the data, despined.

    Panels carry no title; the figure's panel letter and the caption identify
    them, which is what keeps a multi-panel supplement figure readable.
    """
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


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
    assert len(flat) == len(letters), (
        f"{len(flat)} axes but {len(letters)} panel letters"
    )
    fig.canvas.draw()
    inverse = fig.transFigure.inverted()

    # Anchor each letter to its own panel rather than to the top of a row. An
    # earlier version measured the row and offset upward from there, which put
    # the lower letters of a grid into the gap already occupied by the row
    # above's tick labels; how bad the collision looked then depended on the
    # figure's height, so it kept reappearing after resizes.
    for index, ax in enumerate(flat):
        label_extent = ax.yaxis.get_label().get_window_extent()
        axes_extent = ax.get_window_extent()
        # Horizontally: centred on the y-axis label, matching S4 and S5.
        x_fig, _ = inverse.transform((0.5 * (label_extent.x0 + label_extent.x1), 0))
        # Vertically: pinned to this panel's own top edge, so the spacing between
        # rows is the only thing that has to be large enough.
        _, y_fig = inverse.transform((0, axes_extent.y1))
        fig.text(
            x_fig,
            y_fig + 0.008,
            letters[index],
            fontsize=fontsize,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
