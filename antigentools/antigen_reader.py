"""
AntigenReader class and supporting functions for reading and processing antigen simulation outputs.
"""
import glob
import re
import os
import json
import Bio.Phylo as bp
import pandas as pd
import numpy as np


def process_file_path(path: str) -> dict:
    """
    Extracts values from a path and returns them as a dictionary.

    Args:
        path (str): The path string.

    Returns:
        Dict[str, Union[str, float, int]]: A dictionary containing the extracted values.
    """
    components = path.split("/")[1:]
    values = {}

    for component in components:
        if component.count("_") > 1:
            sub_components = split_string(component)
            for sub_component in sub_components:
                key, value = extract_value(sub_component)
                key = convert_to_snake_case(key)
                values[key] = value
        elif "_" not in component:
            # Ignore directories w/o underscores
            continue
        else:
            key, value = extract_value(component)
            key = convert_to_snake_case(key)
            values[key] = value

    return values


def split_string(input_string: str) -> list:
    """
    Helper function that splits a string at underscores to create a list of strings that can be passed into `extract_values()`.

    Args:
        input_string (str): The string to split.

    Returns:
        substrings (Tuple[str]): The list of split strings.
    """
    # Define a regular expression pattern to match substrings with decimals or strings
    pattern = r'[^_]+(?:_[^_]+)?'
    
    # Use re.findall to find all substrings that match the pattern
    substrings = re.findall(pattern, input_string)
    
    return substrings


def extract_value(component: str) -> tuple:
    """
    Extracts the value from a component.

    Args:
        component (str): The component string.

    Returns:
        Union[str, float]: The extracted value(s) as a string, can be a string or float.
    """
    try:
        key, value = component.split("_")
    except ValueError:
        print("ERROR: Invalid component string.")
        return None, None
    try:
        value = float(value)
    except ValueError:
        print(f'ValueError: {value} is not numerical, using as string.')
    return key, value


def convert_to_snake_case(key: str) -> str:
    """
    Converts a camel case string to snake case.

    Args:
        key (str): The camel case string to convert.

    Returns:
        str: The converted snake case string.
    """
    # Convert camel case to snake case
    result = ''
    for char in key:
        if char.isupper():
            result += '_' + char.lower()
        else:
            result += char
    return result.lstrip('_')


def calculate_antigenic_movement_per_year(
    tips_df: pd.DataFrame, date_col: str = "year", phenotype_cols: list = ["ag1", "ag2"]
) -> float:
    """Calculate antigenic movement per year.

    This method splits the tips_df into individual years and calculates the antigenic movement in each year to calculate the average antigenic movement per year and hopefully account for drastic turns in antigenic space.

    Args:
        tips_df (pd.DataFrame): A date-sorted dataframe of tips
        date_col (str, optional): The column name of the date. Defaults to "year".
        phenotype_cols (list, optional): The column names of the antigenic coordinates. Defaults to ["ag1", "ag2"].
    Return:
        antigenic_movement (float): average antigenic movement per year
    """
    # Split up tips_df into individual years
    grouped_tips = [group for _, group in tips_df.groupby(tips_df[date_col].floordiv(1))]
    yearly_antigenic_movement = []
    # Calculate antigenic movement per year
    for small_tips_df in grouped_tips:
        first_virus = small_tips_df.iloc[0]
        last_virus = small_tips_df.iloc[-1]

        # Calculate antigenic movement over sim
        dist = np.linalg.norm(first_virus[phenotype_cols] - last_virus[phenotype_cols])
        yearly_antigenic_movement.append(dist)
    
    antigenic_movement = np.mean(yearly_antigenic_movement)

    return antigenic_movement


def calculate_high_low_epitope_mutation_ratio(tips_df: pd.DataFrame, n_tips: int = 100) -> float:
    """Calculate the ratio of high to low epitope mutations.

    Args:
        tips_df (pd.DataFrame): A dataframe of tips.
        n_tips (int): The number of tips to use in the calculation.

    Returns:
        float: The ratio of high to low epitope mutations.
    """
    if "highEpitopeMutationCount" not in tips_df.columns or "lowEpitopeMutationCount" not in tips_df.columns:
        return None
    if tips_df.highEpitopeMutationCount.sum() == 0 or tips_df.lowEpitopeMutationCount.sum() == 0:
        print("LOG: No high-low epitope mutations found in tips_df. Using standard epitope mutations instead.")
        return None
    # Use last n_tips viruses to calculate epitope mutation ratio
    if n_tips is not None:
        tips_df = tips_df.sort_values(by="year").tail(n_tips)
    high_epitope_mutations = tips_df["highEpitopeMutationCount"].sum()
    low_epitope_mutations = tips_df["lowEpitopeMutationCount"].sum()
    epitope_mutation_ratio = high_epitope_mutations / low_epitope_mutations
    return epitope_mutation_ratio


def calculate_attack_rate(cases: pd.DataFrame) -> pd.DataFrame:
    """Calculate the attack rate for a dataframe of cases.

    Attack rate is defined as the number of infected individuals divided by the number of susceptible individuals.

    Args:
        cases (pd.DataFrame): Dataframe of cases.

    Returns:
        pd.DataFrame: A copy of the dataframe with new columns for total and regional attack rates.
    """
    cases["total_attack_rate"] = cases["totalI"] / cases["totalS"]
    cases["north_attack_rate"] = cases["northI"] / cases["northS"]
    cases["tropics_attack_rate"] = cases["tropicsI"] / cases["tropicsS"]
    cases["south_attack_rate"] = cases["southI"] / cases["southS"]

    return cases


def calculate_case_counts_over_time(cases: pd.DataFrame, population_size: int = 100000) -> pd.DataFrame:
    """Calculate the number of cases per population size for a dataframe of cases.

    Args:
        cases (pd.DataFrame): Dataframe of cases.
        population_size (int, optional): The population size. Defaults to 100000.

    Returns:
        pd.DataFrame: A copy of the dataframe with new columns for total and regional case counts per population size.
    """
    population_prefix = f"{population_size // 1000}k" if population_size > 1000 else population_size

    cases[f"cases_per_{population_prefix}"] = cases["totalCases"] / population_size
    cases[f"north_cases_per_{population_prefix}"] = cases["northCases"] / population_size
    cases[f"tropics_cases_per_{population_prefix}"] = cases["tropicsCases"] / population_size
    cases[f"south_cases_per_{population_prefix}"] = cases["southCases"] / population_size
    return cases


def count_branch_mutations(filename: str, high_low_model: bool=True) -> dict:
    """Parse the branches file and count the number of epitope vs non-epitope mutations on main trunks and side branches.
    Parameters
    ----------
    filename : str
        The path of the branches file.

    Returns
    -------
    mutation_count_dict : dict
        A dictionary with the number of epitope and non-epitope mutations on the trunk and side-branches respectively.
    """
    num_mutations = 0
    difference = 0

    max_e = float("-inf")
    max_e_virus = ""

    max_ne = float("-inf")
    max_ne_virus = ""

    # Store visited sequences we don't double count mutations and mutation counts.
    visited = set()
    trunk_Mutations = np.array([0, 0])
    branch_Mutations = np.array([0, 0])

    # Each line is a sampled virus.
    with open(filename) as f:
        lines = f.readlines()

    counts_dict = {}
    for line in lines:
        elements = line.split(",")
        child = elements[0].split('"')[1]
        v_isTrunk = elements[3]
        
        if not high_low_model:    
            v_eMutations = int(elements[11].strip())
            v_neMutations = int(elements[12].split("}")[0].strip())
            parent = elements[12].split('"')[1]
            vp_isTrunk = elements[15]
            vp_eMutations = int(elements[23].strip())
            vp_neMutations = int(elements[24].split("}")[0].strip())
            v_sequence = elements[8]
            vp_sequence = elements[20]
        else:
            v_eMutations = int(elements[11].strip())
            v_neMutations = int(elements[12])
            v_eLowMutations = int(elements[13].strip())
            v_eHighMutations = int(elements[14].split("}")[0].strip())

            parent = elements[14].split('"')[1]
            vp_isTrunk = elements[17]
            vp_eMutations = int(elements[25].strip())
            vp_neMutations = int(elements[26].strip())
            vp_eLowMutations = int(elements[27].strip())
            vp_eHighMutations = int(elements[28].split("}")[0].strip())

            v_sequence = elements[8]
            vp_sequence = elements[22]

        if v_sequence != vp_sequence:
            num_mutations += 1  # change to hamming distance

        key = (vp_isTrunk, v_isTrunk)

        if key not in counts_dict:
            counts_dict[key] = 1
        else:
            counts_dict[key] += 1

        v_Mutations = np.array([v_eMutations, v_neMutations])
        vp_Mutations = np.array([vp_eMutations, vp_neMutations])
        diff = v_Mutations - vp_Mutations
        if (diff[0] + diff[1]) > 1:
            difference = difference + diff[0] + diff[1] - 1

        # print(vp_Mutations, v_Mutations)
        if key == ("1", "1"):
            if v_eMutations > max_e:
                max_e = v_eMutations
                max_e_virus = child
            if v_neMutations > max_ne:
                max_ne = v_neMutations
                max_ne_virus = child

            trunk_Mutations += diff
        else:
            branch_Mutations += diff
        visited.add(parent)

    mutation_count_dict = {
        "side_branch_epitope_mutations": int(branch_Mutations[0]),
        "side_branch_non-epitope_mutations": int(branch_Mutations[1]),
        "side_branch_epitope_to_non-epitope_ratio": (
            float(branch_Mutations[0] / branch_Mutations[1])
        ),
        "trunk_epitope_mutations": int(trunk_Mutations[0]),
        "trunk_non-epitope_mutations": int(trunk_Mutations[1]),
        "trunk_epitope_to_non-epitope_ratio": float(
            trunk_Mutations[0] / trunk_Mutations[1]
        ),
    }

    return mutation_count_dict


BRANCHES_COLUMNS = [
    'name',
    'date',
    'fitness',
    'trunk',
    'tip',
    'marked',
    'location',
    'layout',
    'sequence',
    'ag1',
    'ag2',
    'n_epitope_mutations',
    'n_non_epitope_mutations',
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
                    antigenic_movement_df = pd.DataFrame({
                        "parameter": "antigenic_movement_per_year",
                        "value": calculate_antigenic_movement_per_year(tips),
                    }, index=[0])
                    epitope_mutation_ratio_df = pd.DataFrame({
                        "parameter": "high_low_epitope_mutation_ratio",
                        "value": calculate_high_low_epitope_mutation_ratio(tips),
                    }, index=[0])
                    summary_file = pd.concat([summary_file, antigenic_movement_df, epitope_mutation_ratio_df], ignore_index=True)

                for key, value in values.items():
                    summary_file[key] = value
                
                summary_file["path"] = path
                self.runs.append(path)
                self.summary_files[path] = summary_file

            except FileNotFoundError:
                #print(f"Failed simulation for directory: {path}")
                failed_sim += 1
            run_counter += 1
            print(f"Progress: {run_counter} / {len(runs)} simulations attempted to be read.")
        self.runs.sort()
        print(f"{failed_sim} / {len(runs)} simulations failed.")


    @staticmethod
    def _read_run(path: str, outdir_path: str="output", file_prefix: str="run-out"):
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
    
    def calculate_branch_stats(self, branch_file_path: str = "simulations/*/*/output/*.branches", high_low_model: bool = True) -> pd.DataFrame:
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
        print(f'LOG: Reading branch files.')
        for branches in branch_files:
            out_name = branches.split('/')[1:3]
            out_name = '/'.join(out_name) + '/' + 'mutation_stats'
            mut_count_dict = count_branch_mutations(branches, high_low_model=high_low_model)
            #branches_df = self._create_branches_df(branches)
            with open('simulations/'+ out_name + '.json', 'w', encoding='utf-8',) as f:
                base_path = '/'.join(branches.split('/')[:-2])
                params = process_file_path(base_path)
                mut_count_dict.update(params)
                # self.set_branches(base_path, branches_df)
                self.set_branches_stats(base_path, mut_count_dict)
                f.write(json.dumps(mut_count_dict))
            n_done +=1
            print(f'LOG: {n_done} / {n_files} branch files processed.')


    def parse_branch_files(self, branch_file_path: str = "simulations/*/*/output/*.branches"):
        """Parse branch files for each run.
        
        Args:
            branch_file_path (str, optional): Path pattern to locate the branch files. Defaults to "simulations/*/*/output/*.branches".

        Returns:
            None
        """
        branch_files = glob.glob(branch_file_path)
        n_files = len(branch_files)
        n_done = 0
        print(f'LOG: Reading branch files.')
        for branches in branch_files:
            out_name = branches.split('/')[1:3]
            out_name = '/'.join(out_name) + '/' + 'mutation_stats'
            mut_count_dict = count_branch_mutations(branches)
            branches_df = self._create_branches_df(branches)
            with open('simulations/'+ out_name + '.json', 'w', encoding='utf-8',) as f:
                base_path = '/'.join(branches.split('/')[:-2])
                params = process_file_path(base_path)
                mut_count_dict.update(params)
                self.set_branches(base_path, branches_df)
                self.set_branches_stats(base_path, mut_count_dict)
                f.write(json.dumps(mut_count_dict))
            n_done +=1
            complete_percent = n_done / n_files
            print(f'LOG: {n_done}/{n_files} branch files read: {round(complete_percent,2) * 100}% complete')

    
    def aggregate_branch_stats(self, input_path = "simulations/*/*/mutation_stats.json", output_path = "mutation_stats.csv"):
        """Aggregate branch stats into a single dataframe and write to csv.

        Args:
            input_path (str, optional): Path pattern to locate the mutation stats json files. Defaults to "simulations/*/*/mutation_stats.json".
            output_path (str, optional): Path to write the aggregated mutation stats csv file. Defaults to "simulations/mutation_stats.csv".
        """
        files = glob.glob(input_path)
        dfs = []
        for file in files:
            with open(file, 'r') as f:
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
        with open(branches_path, 'r') as f:
            branches_lines = f.readlines()
        
        bad_chars = '{} "" \n'
        rgx = re.compile('[%s]' % bad_chars)
        entry_list = []
        # Step one: loop through branches lines and perform cleanup
        for line in branches_lines:
            text = rgx.sub('', line)
            text = text.split('\t')
            child = text[0]
            parent = text[1]
            child_dict = dict(zip(BRANCHES_COLUMNS, child.split(',')))
            parent_dict = dict(zip(BRANCHES_COLUMNS, parent.split(',')))
            entry_list.append(child_dict)
            entry_list.append(parent_dict)
        branches_df = pd.DataFrame(entry_list)
        branches_df = branches_df.apply(pd.to_numeric, errors='ignore')
        # Drop duplicates
        branches_df_clean = branches_df.drop_duplicates(subset=['name', 'date'], keep='last', inplace=False)
        return branches_df_clean
    

    def write_tips_to_fasta(self, dataframe: pd.DataFrame, output_path: str, write_metadata: bool = False, write_details_to_fasta: bool = False, variant_col: str = 'variant') -> None:
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
        with open(output_path, 'w') as fasta_file:
            if write_metadata:
                metadata = dataframe[['name', 'year', 'country', variant_col, 'fitness']]  # Select relevant columns
                # Rename columns to work with nextstrain
                metadata = metadata.rename(columns={'name': 'strain', variant_col: 'clade_membership'})
                metadata.to_csv(output_path.replace('.fasta', '_metadata.tsv'), sep='\t', index=False)

            for index, row in dataframe.iterrows():
                sequence = row['nucleotideSequence']
                name = row['name']
                date = row['year']
                fitness = row['fitness']
                # Write metadata to FASTA header
                if write_details_to_fasta:
                    fasta_file.write(f'>{name}|{date}|{fitness}\n')
                else:
                    fasta_file.write(f'>{name}\n')
                fasta_file.write(f'{sequence}\n')
    
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
        if host_string == '\n':
            return host_memory
        host_string = host_string.strip()
        # If there are multiple entries, split them [this means we have ')()']
        if host_string.count(')(') > 0:
            host_string = host_string.split(')(')
            # Remove the parentheses and split the string on commas
            host_string = [host.replace('(', '').replace(')', '') for host in host_string]
            host_memory = [host.split(',') for host in host_string]
            # Make entries in sublists floats
            host_memory = [[float(coord) for coord in infection] for infection in host_memory]
        else:
            # Remove leading and trailing parentheses
            host_string = host_string[1:-1]
            host_memory = host_string.split(',')
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
                if 'date' in line:
                    date = line.split()[1]
                    host_memories[date] = {}
                elif 'contactRate' in line:
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


def read_model_estimates(paths: list) -> pd.DataFrame:
    """ Read evofr forecasting estimates from a list of paths.

    Parses the model, location, and pivot date from the path to add to the dataframe.

    Parameters:
    ---------------
        paths (list): List of paths to read

    Returns:
    ---------------
        pd.DataFrame: Dataframe of estimates
    """
    dfs = []
    for path in paths:
        model = path.split("/")[-3]
        deme = path.split("/")[-2]
        pivot_date = path.split("/")[-1].split("_")[-1].split(".")[0]
        df = pd.read_csv(path, sep="\t")
        df['model'] = model
        df['location'] = deme
        df['pivot_date'] = pivot_date
        dfs.append(df)
    return pd.concat(dfs)