"""Tests for antigentools/supplement_style.py.

Only the parts that silently change how a published figure looks are covered.
``hollow_boxes`` is the interesting one: it was copy-pasted into four figures
(S1/S2, S3, S4, S5, figure 4/S6) before being promoted here, and the copies had
already begun to differ -- S3 and S4 drew flat grey boxes while the rest drew
coloured outlines. A regression would not raise; it would just put a grey slab
over the point cloud in whichever figure lost the call.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from antigentools import supplement_style  # noqa: E402


@pytest.fixture
def boxed_axes():
    """An axes carrying two filled boxes in known colours."""
    fig, ax = plt.subplots()
    ax.boxplot([[1, 2, 3, 4], [2, 3, 4, 5]], patch_artist=True)
    colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color)
    yield ax, colors
    plt.close(fig)


class TestHollowBoxes:
    def test_face_is_cleared_and_colour_moves_to_the_edge(self, boxed_axes):
        ax, colors = boxed_axes
        supplement_style.hollow_boxes(ax)
        for patch, color in zip(ax.patches, colors):
            # Alpha 0 is matplotlib's representation of facecolor "none".
            assert patch.get_facecolor()[3] == 0.0
            assert patch.get_edgecolor() == color

    def test_each_box_keeps_its_own_colour(self, boxed_axes):
        """A loop bug that reused one colour would make every box identical."""
        ax, _ = boxed_axes
        supplement_style.hollow_boxes(ax)
        edges = [patch.get_edgecolor() for patch in ax.patches]
        assert len(set(edges)) == len(edges)

    def test_linewidth_is_applied(self, boxed_axes):
        ax, _ = boxed_axes
        supplement_style.hollow_boxes(ax, linewidth=3.5)
        assert all(patch.get_linewidth() == 3.5 for patch in ax.patches)

    def test_default_linewidth_matches_the_published_figures(self):
        """S5 panel A and figure 4 panel A were both drawn at 2.0."""
        import inspect

        default = (
            inspect.signature(supplement_style.hollow_boxes)
            .parameters["linewidth"]
            .default
        )
        assert default == 2.0

    def test_is_idempotent(self, boxed_axes):
        """Calling twice must not turn the edges transparent.

        The second pass reads the (now cleared) facecolor, so a naive
        implementation would copy alpha 0 onto the edge and erase the outline.
        """
        ax, colors = boxed_axes
        supplement_style.hollow_boxes(ax)
        supplement_style.hollow_boxes(ax)
        for patch, color in zip(ax.patches, colors):
            assert patch.get_edgecolor() == color


class TestPanelStyling:
    def test_grid_is_horizontal_only(self):
        """Vertical gridlines over a categorical axis are texture, not information."""
        fig, ax = plt.subplots()
        supplement_style.style_panel(ax)
        assert ax.yaxis.get_gridlines()[0].get_visible()
        assert not ax.xaxis.get_gridlines()[0].get_visible()
        assert ax.get_axisbelow()
        plt.close(fig)

    def test_top_and_right_spines_are_hidden(self):
        fig, ax = plt.subplots()
        supplement_style.style_panel(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
        plt.close(fig)


class TestJitterSeed:
    def test_seed_is_the_documented_value(self):
        """Changing this moves every point in every strip overlay."""
        assert supplement_style.JITTER_SEED == 5

    def test_seeding_makes_the_next_draw_reproducible(self):
        import numpy as np

        supplement_style.seed_jitter()
        first = np.random.rand(5)
        supplement_style.seed_jitter()
        assert np.array_equal(first, np.random.rand(5))
