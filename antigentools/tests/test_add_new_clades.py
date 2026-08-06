"""Unit tests for scripts/add_new_clades.py site weighting.

Regression cover for a bug where the per-site weights in
configs/weights_per_site_for_clades.json were silently inert: JSON object keys
are strings, `score` looked sites up by integer position, so `pos in w` never
matched and every mutation fell through to the per-CDS default. The clade
assignment ran effectively unweighted while appearing to be configured with the
epitope sites.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "add_new_clades.py"
WEIGHTS_PATH = REPO_ROOT / "configs" / "weights_per_site_for_clades.json"


@pytest.fixture(scope="module")
def add_new_clades():
    """Import add_new_clades.py as a module."""
    spec = importlib.util.spec_from_file_location("add_new_clades", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["add_new_clades"] = module
    spec.loader.exec_module(module)
    return module


def _node(mutations):
    return {
        "bushiness": 1.0,
        "node_attrs": {},
        "branch_attrs": {"mutations": {"HA1": mutations}},
    }


class TestCoerceSiteWeightKeys:
    def test_site_keys_become_ints_and_default_survives(self, add_new_clades):
        out = add_new_clades.coerce_site_weight_keys(
            {"HA1": {"default": 1, "145": 2, "155": 2}}
        )
        assert out["HA1"][145] == 2
        assert out["HA1"][155] == 2
        assert out["HA1"]["default"] == 1
        assert "145" not in out["HA1"]

    def test_rejects_a_key_that_is_neither_default_nor_a_site(self, add_new_clades):
        with pytest.raises(AssertionError, match="neither 'default' nor a site"):
            add_new_clades.coerce_site_weight_keys({"HA1": {"epitope": 2}})

    def test_shipped_weights_file_carries_the_epitope_sites(self, add_new_clades):
        with open(WEIGHTS_PATH) as fh:
            raw = json.load(fh)
        out = add_new_clades.coerce_site_weight_keys(raw["h3n2"])
        weighted = [k for k in out["HA1"] if k != "default"]
        # The 49 Luksza & Lassig epitope sites the manuscript reports.
        assert len(weighted) == 49
        assert all(isinstance(k, int) for k in weighted)
        assert all(out["HA1"][k] > out["HA1"]["default"] for k in weighted)


class TestScoreUsesSiteWeights:
    def test_epitope_mutation_outscores_non_epitope(self, add_new_clades):
        weights = add_new_clades.coerce_site_weight_keys(
            {"HA1": {"default": 1, "145": 2}}
        )
        epitope = add_new_clades.score(
            _node(["N145K"]), weights=weights, proteins=["HA1"]
        )
        non_epitope = add_new_clades.score(
            _node(["A400T"]), weights=weights, proteins=["HA1"]
        )
        assert epitope > non_epitope

    def test_string_keyed_weights_are_inert_without_coercion(self, add_new_clades):
        """The original bug: uncoerced weights score an epitope site as default."""
        raw = {"HA1": {"default": 1, "145": 2}}
        epitope = add_new_clades.score(_node(["N145K"]), weights=raw, proteins=["HA1"])
        non_epitope = add_new_clades.score(
            _node(["A400T"]), weights=raw, proteins=["HA1"]
        )
        assert epitope == non_epitope
