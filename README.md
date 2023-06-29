# Flu-forecasting with data simulated by `antigen-prime`

## Introduction

In this analysis, we run the forecasting pipeline described in John's paper.
Instead of using `SANTA-SIM` as our sequence simulator, we instead use `antigen-prime`, a modified version of the epidemological simulator `antigen` that supports sequence evolution.

## Quickstart

## Notes

The following things should be addressed later:
- The pipeline is made for SANTA-SIM data, and `antigen-prime` doesn't support crucial things like fitness.
    - For now, I have assigned viruses random fitness values in the pipeline (this is done in `scripts/standardize_simulated_sequence_data.py`) if a dataframe of sequence metadata is missing fitness values.
    - We could obviously do something a bit smarter -- but since `antigen` does not have any baked-in notion of fitness, we may need to get creative.
- To run the pipeline on simulated sequences only (this is configured in `config/minimal_config.json`):

        snakemake -p --cores 8 --use-conda --conda-frontend mamba --configfile config/minimal_config.json