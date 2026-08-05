"""Tests for antigentools/growth_rate_panels.py.

These are regression guards for the specific ways this module can silently break
the two published figures it draws (main figure 4 and supplement S6), not tests of
rendering — the repo has no rendering tests and this is not the place to start one.

Each case pins a failure mode that actually occurred or was one edit away:

* the variant-inclusion thresholds were free variables in the notebooks, and the
  function they feed has *different* defaults, so losing them would quietly change
  both figures;
* ``plot_variant_incidence`` already exists in ``antigentools.plot`` as a
  whole-figure function, so the axes-level one here must not shadow it;
* the font sizes must stay tied to the shared supplement convention rather than
  drifting back to hardcoded numbers.

Note the figures themselves are *not* reproducible run to run: seaborn's
``stripplot`` jitters points from the global RNG and takes no seed, so two
consecutive renders of identical code differ. Any future attempt to verify a
change here by comparing images has to seed ``numpy.random`` first.
"""

from __future__ import annotations

import inspect

import pytest

from antigentools import growth_rate_panels as grp
from antigentools import plot as legacy_plot
from antigentools.supplement_style import scale_for_canvas


class TestPublicSurface:
    def test_exposes_the_promoted_panels(self):
        """All four panel helpers plus the driver moved out of the notebooks."""
        expected = {
            "plot_observed_cases_clean",
            "plot_frequencies_nowcast_only",
            "plot_variant_incidence_panel",
            "plot_r_data_vs_r_model_clean",
            "plot_combined_growth_rate_figure",
        }
        actual = {
            name
            for name, _ in inspect.getmembers(grp, inspect.isfunction)
            if name.startswith("plot_")
        }
        assert expected <= actual

    def test_panels_are_axes_level(self):
        """Every panel helper takes the axes to draw on as its first argument."""
        for name in (
            "plot_observed_cases_clean",
            "plot_frequencies_nowcast_only",
            "plot_variant_incidence_panel",
            "plot_r_data_vs_r_model_clean",
        ):
            first = next(iter(inspect.signature(getattr(grp, name)).parameters))
            assert first == "ax", f"{name} takes {first!r} first, expected 'ax'"


class TestNameCollision:
    """``antigentools.plot`` already had a ``plot_variant_incidence``."""

    def test_legacy_whole_figure_version_is_untouched(self):
        # The legacy one builds its own figure, so its first parameter is data.
        first = next(
            iter(inspect.signature(legacy_plot.plot_variant_incidence).parameters)
        )
        assert first == "growth_rates_df"

    def test_promoted_version_is_renamed(self):
        """The axes-level helper must not be named plot_variant_incidence.

        Adding it to plot.py under the original name would shadow the whole-figure
        function and break plot_growth_rate_dynamics, which calls it.
        """
        assert not hasattr(grp, "plot_variant_incidence")
        assert hasattr(grp, "plot_variant_incidence_panel")


class TestFigureCriticalConstants:
    def test_thresholds_match_what_the_figures_were_drawn_with(self):
        """These are NOT the defaults of the function they are passed to.

        ``get_filtered_growth_rates_df`` defaults to ``min_sequence_count=5`` and
        ``min_variant_frequency=0.05``. The notebooks overrode both, so dropping
        these constants in favour of the callee's defaults would change which
        variants appear in every panel of both published figures.
        """
        assert grp.MIN_SEQUENCE_COUNT == 10
        assert grp.MIN_VARIANT_FREQUENCY == 0.01
        assert grp.MIN_VARIANT_INCIDENCE == 50.0

    def test_thresholds_differ_from_the_callee_defaults(self):
        """Guard the reason the constants exist, not just their values."""
        callee = inspect.signature(grp.get_filtered_growth_rates_df).parameters
        assert callee["min_sequence_count"].default != grp.MIN_SEQUENCE_COUNT
        assert callee["min_variant_frequency"].default != grp.MIN_VARIANT_FREQUENCY

    def test_font_sizes_come_from_the_shared_convention(self):
        """Sizes must track supplement_style, not drift back to literals."""
        assert grp.DEFAULT_CANVAS_WIDTH_IN == 30.0
        assert grp.DEFAULT_SIZES == scale_for_canvas(30.0)

    def test_deme_palette_matches_the_other_figures(self):
        assert grp.DEME_PALETTE == {
            "tropics": "#3498db",
            "north": "#e67e22",
            "south": "#2ecc71",
        }


class TestDriverDefaults:
    def test_figure_level_title_is_off(self):
        """A figure-level title duplicates the caption; both figures dropped it."""
        default = (
            inspect.signature(grp.plot_combined_growth_rate_figure)
            .parameters["show_main_title"]
            .default
        )
        assert default is False

    def test_panel_font_defaults_are_the_scaled_sizes(self):
        params = inspect.signature(grp.plot_combined_growth_rate_figure).parameters
        assert params["label_fontsize"].default == grp.DEFAULT_SIZES["label"]
        assert params["tick_fontsize"].default == grp.DEFAULT_SIZES["tick"]
        assert params["panel_label_size"].default == grp.DEFAULT_SIZES["panel_letter"]


@pytest.mark.parametrize(
    "canvas,expected_label",
    [(11.0, 14), (12.0, 15), (30.0, 38)],
)
def test_canvas_scaling_reproduces_the_hardcoded_sizes(canvas, expected_label):
    """The canvases in use land on the sizes the notebooks had hardcoded.

    ``variant-zoom-in-finder`` draws at 11 inches and used 14; the aggregated
    supplement figures use the 12-inch reference and 15; figure 4 and S6 draw at
    30 inches. This is what let the notebooks adopt the shared scaling without
    any of them changing appearance.
    """
    assert scale_for_canvas(canvas)["label"] == expected_label
