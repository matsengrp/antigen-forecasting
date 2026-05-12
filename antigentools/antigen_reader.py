import glob
import re
import json
import numpy as np
import Bio.Phylo as bp
import pandas as pd


BRANCHES_COLUMNS = [
    "name",
    "date",
    "fitness",
    "trunk",
    "tip",
    "marked",
    "location",
    "layout",
    "sequence",
    "ag1",
    "ag2",
    "n_epitope_mutations",
    "n_non_epitope_mutations",
]


class AntigenReader:
    def __init__(self):
        self.runs = []
        self.cases = {}
        self.trees = {}
        self.tips = {}
        self.branches = {}
        self.summary_files = {}
        self.branches_stats = {}

    def read_runs(
        self,
        path: str = "simulations/*/*",
        outdir_path: str = "output",
        file_prefix: str = "run-out",
        verbose: bool = False,
    ):
        """Read and process simulation data from runs.

        Args:
            path (str, optional): Path pattern to locate the simulation runs. Defaults to "simulations/*/*".
            outdir_path (str, optional): Subdirectory of the run path where output files are located. Defaults to "output".
            file_prefix (str, optional): File prefix for tree and tips files. Defaults to "run-out".
            verbose (bool, optional): Loads dataframes as well, adds antigenic movement column. Defaults to False.
        """
        runs = glob.glob(path)
        run_counter = 0
        failed_sim = 0
        print(f"Reading {len(runs)} simulations.")
        for path in runs:
            try:
                # Parse file path.
                values = process_file_path(path)
                # Read summary file.
                summary_file = pd.read_csv(
                    path + "/out.summary",
                    sep="\t",
                    skiprows=1,
                    names=["parameter", "value"],
                )

                if verbose:
                    cases, tree, tips = self._read_run(path, outdir_path, file_prefix)
                    self.cases[path] = cases
                    self.trees[path] = tree
                    self.tips[path] = tips
                    # Calculate attack rate
                    cases = calculate_attack_rate(cases)
                    # Calculate case counts over time
                    cases = calculate_case_counts_over_time(cases)
                    # Add antigenic movement column
                    antigenic_movement_df = pd.DataFrame(
                        {
                            "parameter": "antigenic_movement_per_year",
                            "value": calculate_antigenic_movement_per_year(tips),
                        },
                        index=[0],
                    )
                    epitope_mutation_ratio_df = pd.DataFrame(
                        {
                            "parameter": "high_low_epitope_mutation_ratio",
                            "value": calculate_high_low_epitope_mutation_ratio(tips),
                        },
                        index=[0],
                    )
                    summary_file = pd.concat(
                        [
                            summary_file,
                            antigenic_movement_df,
                            epitope_mutation_ratio_df,
                        ],
                        ignore_index=True,
                    )

                for key, value in values.items():
                    summary_file[key] = value

                summary_file["path"] = path
                self.runs.append(path)
                self.summary_files[path] = summary_file

            except FileNotFoundError:
                # print(f"Failed simulation for directory: {path}")
                failed_sim += 1
            run_counter += 1
            print(
                f"Progress: {run_counter} / {len(runs)} simulations attempted to be read."
            )
        self.runs.sort()
        print(f"{failed_sim} / {len(runs)} simulations failed.")

    @staticmethod
    def _read_run(path: str, outdir_path: str = "output", file_prefix: str = "run-out"):
        cases = pd.read_csv(f"{path}/out_timeseries.csv")
        try:
            tree = bp.read(f"{path}/{outdir_path}/{file_prefix}.trees", "newick")
        except:
            print(f"Failed to read tree for {path}")
            tree = None
        tips = pd.read_csv(f"{path}/{outdir_path}/{file_prefix}.tips")
        cases["path"] = path
        tips["path"] = path

        cases = calculate_case_counts_over_time(cases)
        cases = calculate_attack_rate(cases)

        return cases, tree, tips

    def calculate_branch_stats(
        self,
        branch_file_path: str = "simulations/*/*/output/*.branches",
        high_low_model: bool = True,
    ) -> pd.DataFrame:
        """Read branch files and count mutation types along the tree.

        Args:
            branch_file_path (str, optional): Path pattern to locate the branch files. Defaults to "simulations/*/*/output/*.branches".
            high_low_model (bool, optional): Whether accomodate extra columns produced by the high-low model. Defaults to False.

        Returns
            pd.DataFrame: DataFrame with mutation counts for each run.
        """
        branch_files = glob.glob(branch_file_path)
        n_files = len(branch_files)
        n_done = 0
        print("LOG: Reading branch files.")
        for branches in branch_files:
            out_name = branches.split("/")[1:3]
            out_name = "/".join(out_name) + "/" + "mutation_stats"
            mut_count_dict = count_branch_mutations(
                branches, high_low_model=high_low_model
            )
            # branches_df = self._create_branches_df(branches)
            with open(
                "simulations/" + out_name + ".json",
                "w",
                encoding="utf-8",
            ) as f:
                base_path = "/".join(branches.split("/")[:-2])
                params = process_file_path(base_path)
                mut_count_dict.update(params)
                # self.set_branches(base_path, branches_df)
                self.set_branches_stats(base_path, mut_count_dict)
                f.write(json.dumps(mut_count_dict))
            n_done += 1
            print(f"LOG: {n_done} / {n_files} branch files processed.")

    def parse_branch_files(
        self, branch_file_path: str = "simulations/*/*/output/*.branches"
    ):
        """Parse branch files for each run.

        Args:
            branch_file_path (str, optional): Path pattern to locate the branch files. Defaults to "simulations/*/*/output/*.branches".

        Returns:
            None
        """
        branch_files = glob.glob(branch_file_path)
        n_files = len(branch_files)
        n_done = 0
        print("LOG: Reading branch files.")
        for branches in branch_files:
            out_name = branches.split("/")[1:3]
            out_name = "/".join(out_name) + "/" + "mutation_stats"
            mut_count_dict = count_branch_mutations(branches)
            branches_df = self._create_branches_df(branches)
            with open(
                "simulations/" + out_name + ".json",
                "w",
                encoding="utf-8",
            ) as f:
                base_path = "/".join(branches.split("/")[:-2])
                params = process_file_path(base_path)
                mut_count_dict.update(params)
                self.set_branches(base_path, branches_df)
                self.set_branches_stats(base_path, mut_count_dict)
                f.write(json.dumps(mut_count_dict))
            n_done += 1
            complete_percent = n_done / n_files
            print(
                f"LOG: {n_done}/{n_files} branch files read: {round(complete_percent, 2) * 100}% complete"
            )

    def aggregate_branch_stats(
        self,
        input_path="simulations/*/*/mutation_stats.json",
        output_path="mutation_stats.csv",
    ):
        """Aggregate branch stats into a single dataframe and write to csv.

        Args:
            input_path (str, optional): Path pattern to locate the mutation stats json files. Defaults to "simulations/*/*/mutation_stats.json".
            output_path (str, optional): Path to write the aggregated mutation stats csv file. Defaults to "simulations/mutation_stats.csv".
        """
        files = glob.glob(input_path)
        dfs = []
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
            df = pd.DataFrame([data])
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(output_path, index=False)

    def _create_branches_df(self, branches_path: str) -> pd.DataFrame:
        """Create a dataframe from the branches dictionary.

        Args:
            branches_file (list): List of branches file lines.

        Returns:
            pd.DataFrame: Branches dataframe.
        """
        with open(branches_path, "r") as f:
            branches_lines = f.readlines()

        bad_chars = '{} "" \n'
        rgx = re.compile("[%s]" % bad_chars)
        entry_list = []
        # Step one: loop through branches lines and perform cleanup
        for line in branches_lines:
            text = rgx.sub("", line)
            text = text.split("\t")
            child = text[0]
            parent = text[1]
            child_dict = dict(zip(BRANCHES_COLUMNS, child.split(",")))
            parent_dict = dict(zip(BRANCHES_COLUMNS, parent.split(",")))
            entry_list.append(child_dict)
            entry_list.append(parent_dict)
        branches_df = pd.DataFrame(entry_list)
        branches_df = branches_df.apply(pd.to_numeric, errors="ignore")
        # Drop duplicates
        branches_df_clean = branches_df.drop_duplicates(
            subset=["name", "date"], keep="last", inplace=False
        )
        return branches_df_clean

    def write_tips_to_fasta(
        self,
        dataframe: pd.DataFrame,
        output_path: str,
        write_metadata: bool = False,
        write_details_to_fasta: bool = False,
        variant_col: str = "variant",
    ) -> None:
        """
        Write DataFrame contents to a FASTA file with metadata.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing 'name', 'year', 'fitness', and optionally 'variant' columns.
            output_path (str): The path to the output FASTA file.
            write_metadata (bool): Whether to write metadata to a metadata.tsv file. Default is False.
            write_details_to_fasta (bool): Whether to write detailed metadata to the FASTA header. Default is False.
            variant_col (str): The column name for the variant column. Default is 'variant'.

        Returns:
            None
        """
        with open(output_path, "w") as fasta_file:
            if write_metadata:
                metadata = dataframe[
                    ["name", "year", "country", variant_col, "fitness"]
                ]  # Select relevant columns
                # Rename columns to work with nextstrain
                metadata = metadata.rename(
                    columns={"name": "strain", variant_col: "clade_membership"}
                )
                metadata.to_csv(
                    output_path.replace(".fasta", "_metadata.tsv"),
                    sep="\t",
                    index=False,
                )

            for index, row in dataframe.iterrows():
                sequence = row["nucleotideSequence"]
                name = row["name"]
                date = row["year"]
                # Write metadata to FASTA header
                if write_details_to_fasta:
                    fasta_file.write(f">{name}|{date}\n")
                else:
                    fasta_file.write(f">{name}\n")
                fasta_file.write(f"{sequence}\n")

    def get_cases(self, path: str) -> pd.DataFrame:
        return self.cases.get(path)

    def get_tree(self, path: str) -> bp.BaseTree.Tree:
        return self.trees.get(path)

    def get_tips(self, path: str) -> pd.DataFrame:
        return self.tips.get(path)

    def get_summary_file(self, path: str) -> pd.DataFrame:
        return self.summary_files.get(path)

    def get_branches(self, path: str) -> pd.DataFrame:
        return self.branches.get(path)

    def get_branches_stats(self, path: str) -> dict:
        return self.branches_stats.get(path)

    def set_cases(self, path: str, cases: pd.DataFrame):
        self.cases[path] = cases

    def set_tree(self, path: str, tree: bp.BaseTree.Tree):
        self.trees[path] = tree

    def set_tips(self, path: str, tips: pd.DataFrame):
        self.tips[path] = tips

    def set_summary_file(self, path: str, summary_file: pd.DataFrame):
        self.summary_files[path] = summary_file

    def set_branches(self, path: str, branches: pd.DataFrame):
        self.branches[path] = branches

    def set_branches_stats(self, path: str, branches_stats: dict):
        self.branches_stats[path] = branches_stats

    def parse_host(self, host_string: str) -> list:
        """
        Parse the host string into a numpy array.

        Parameters
        ----------
        host_string : str
            The string representation of the host.

        Returns
        -------
        host_memory : list
            The host as a list of coordinates.
        """
        host_memory = []
        # Check for naive host
        if host_string == "\n":
            return host_memory
        host_string = host_string.strip()
        # If there are multiple entries, split them [this means we have ')()']
        if host_string.count(")(") > 0:
            host_string = host_string.split(")(")
            # Remove the parentheses and split the string on commas
            host_string = [
                host.replace("(", "").replace(")", "") for host in host_string
            ]
            host_memory = [host.split(",") for host in host_string]
            # Make entries in sublists floats
            host_memory = [
                [float(coord) for coord in infection] for infection in host_memory
            ]
        else:
            # Remove leading and trailing parentheses
            host_string = host_string[1:-1]
            host_memory = host_string.split(",")
            # Make entries floats
            host_memory = [[float(coord) for coord in host_memory]]
        return host_memory

    def load_memories(self, memory_path: str) -> tuple:
        """
        Load the immune memory file and return a dictionary of host memories.

        Parameters
        ----------
        memory_path : str
            The path to the immune memory file.

        Returns
        -------
        tuple : (dict, dict)
            A tuple of host memories and contact rates.
        """
        host_memories = {}
        contact_rates = {}
        with open(memory_path, "r") as f:
            for line in f:
                if "date" in line:
                    date = line.split()[1]
                    host_memories[date] = {}
                elif "contactRate" in line:
                    contact_rate = line.split()[1]
                    contact_rate[date] = {}
                else:
                    deme, sample = line.split(sep=":")
                    if deme not in host_memories[date]:
                        host_memories[date][deme] = []
                    if deme not in contact_rates[date]:
                        contact_rates[date][deme] = float(contact_rate)
                    memory = self.parse_host(sample)
                    host_memories[date][deme].append(memory)
        return (host_memories, contact_rates)


# Note: The following functions still need to be imported from the original module
# or implemented separately:
# - process_file_path
# - calculate_attack_rate
# - calculate_case_counts_over_time
# - calculate_high_low_epitope_mutation_ratio


def count_branch_mutations(
    branches_path: str, high_low_model: bool = True
) -> dict[str, float]:
    """Count trunk and side-branch epitope/non-epitope mutations from a .branches file.

    Args:
        branches_path: Path to the ``run-out.branches`` file.
        high_low_model: Accepted for API compatibility; currently unused.

    Returns:
        Dict with keys: ``trunk_epitope_mutations``, ``trunk_non_epitope_mutations``,
        ``trunk_epitope_to_non-epitope_ratio``, ``side_branch_epitope_mutations``,
        ``side_branch_non_epitope_mutations``, ``side_branch_epitope_to_non-epitope_ratio``.
        Ratio is ``float("nan")`` when the denominator is zero.

    Raises:
        FileNotFoundError: If ``branches_path`` does not exist.
    """
    from pathlib import Path as _Path

    path = _Path(branches_path)
    if not path.exists():
        raise FileNotFoundError(f"Branches file not found: {path}")

    with open(path, "r") as f:
        branches_lines = f.readlines()

    bad_chars = '{} "" \n'
    rgx = re.compile("[%s]" % bad_chars)
    entry_list = []
    for line in branches_lines:
        text = rgx.sub("", line)
        parts = text.split("\t")
        if len(parts) < 2:
            continue
        child = parts[0]
        parent = parts[1]
        for raw in (child, parent):
            values = raw.split(",")
            if len(values) == len(BRANCHES_COLUMNS):
                entry_list.append(dict(zip(BRANCHES_COLUMNS, values)))

    df = pd.DataFrame(entry_list)
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.drop_duplicates(subset=["name", "date"], keep="last")

    trunk = df[df["trunk"] == 1]
    side = df[df["trunk"] != 1]

    def _ratio(num: float, den: float) -> float:
        return float("nan") if den == 0 else num / den

    trunk_epi = float(trunk["n_epitope_mutations"].sum())
    trunk_non = float(trunk["n_non_epitope_mutations"].sum())
    side_epi = float(side["n_epitope_mutations"].sum())
    side_non = float(side["n_non_epitope_mutations"].sum())

    return {
        "trunk_epitope_mutations": trunk_epi,
        "trunk_non_epitope_mutations": trunk_non,
        "trunk_epitope_to_non-epitope_ratio": _ratio(trunk_epi, trunk_non),
        "side_branch_epitope_mutations": side_epi,
        "side_branch_non_epitope_mutations": side_non,
        "side_branch_epitope_to_non-epitope_ratio": _ratio(side_epi, side_non),
    }


def calculate_antigenic_movement_per_year(
    tips_df: pd.DataFrame,
    date_col: str = "year",
    phenotype_cols: list[str] | None = None,
    window_size: float = 1.0,
) -> float:
    """Calculate antigenic movement per year using sliding windows.

    Creates overlapping time windows of specified size, calculates antigenic
    movement within each window, then normalizes by window duration to get
    movement per year.

    Args:
        tips_df: Date-sorted DataFrame of tips with antigenic coordinates.
        date_col: Column name of the date. Defaults to ``"year"``.
        phenotype_cols: Column names of antigenic coordinates.
            Defaults to ``["ag1", "ag2"]``.
        window_size: Size of time window in years. Defaults to ``1.0``.

    Returns:
        Average antigenic movement per year.
    """
    if phenotype_cols is None:
        phenotype_cols = ["ag1", "ag2"]

    if len(tips_df) < 2:
        return 0.0

    tips_df = tips_df.sort_values(date_col).reset_index(drop=True)

    min_time = tips_df[date_col].min()
    max_time = tips_df[date_col].max()
    total_duration = max_time - min_time

    if total_duration <= 0:
        return 0.0

    window_movements = []
    current_start = min_time
    while current_start + window_size <= max_time:
        window_end = current_start + window_size
        window_mask = (tips_df[date_col] >= current_start) & (
            tips_df[date_col] <= window_end
        )
        window_tips = tips_df[window_mask]

        if len(window_tips) >= 2:
            first_virus = window_tips.iloc[0]
            last_virus = window_tips.iloc[-1]
            dist = np.linalg.norm(
                first_virus[phenotype_cols].to_numpy()
                - last_virus[phenotype_cols].to_numpy()
            )
            actual_duration = last_virus[date_col] - first_virus[date_col]
            if actual_duration > 0:
                window_movements.append(dist / actual_duration)

        current_start += window_size / 2

    if not window_movements:
        first_virus = tips_df.iloc[0]
        last_virus = tips_df.iloc[-1]
        total_dist = np.linalg.norm(
            first_virus[phenotype_cols].to_numpy()
            - last_virus[phenotype_cols].to_numpy()
        )
        return float(total_dist / total_duration)

    return float(np.mean(window_movements))
