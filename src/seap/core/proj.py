import pandas as pd

# This script is used to generate a projection string for Wannier90.
# It reads two CSV files: 'center.csv' and 'orbital.csv', which contain information about the centers and orbitals of the Wannier functions.
# The script then merges the two DataFrames on the 'band_index' and 'molecule_index' columns.
# The merged DataFrame is used to generate a projection string, which is written to 'proj.out'.

def main():
    """
    Main function to process CSV files and generate projection string.
    
    Returns
    -------
    None
    """
    # Read the 'center.csv' file into a DataFrame 'df1'
    df1 = pd.read_csv('center.csv')
    
    # Read the 'orbital.csv' file into a DataFrame 'df2'
    df2 = pd.read_csv('orbital.csv')

    # The following block of code is commented out and not currently in use.
    # It was intended to process 'df2' by calculating the absolute value of 'cval',
    # grouping by 'band_index', and sorting each group by 'abs_cval' in descending order.
    # The goal was to select unique combinations of 'molecule_index' and 'orbital'
    # for each 'band_index', ensuring no duplicates in 'selected_combo'.
    """
    df2['abs_cval'] = df2['cval'].abs()
    df2_grouped = df2.groupby('band_index', as_index=False).apply(lambda x: x.sort_values('abs_cval', ascending=False))
    selected_combo = set()
    selected_data = []
    for band_index, group in df2_grouped.groupby('band_index'):
        selected_row = group.iloc[0]
        combo = (selected_row['molecule_index'], selected_row['orbital'])
        if combo not in selected_combo:
            selected_data.append(selected_row)
            selected_combo.add(combo)
        else:
            for _, row in group.iloc[1:].iterrows():
                combo = (row['molecule_index'], row['orbital'])
                if combo not in selected_combo:
                    selected_data.append(row)
                    selected_combo.add(combo)
                    break
    df3 = pd.DataFrame(selected_data)
    merged_df = pd.merge(df1, df3, on=['band_index', 'molecule_index'])
    """

    # Merge 'df1' and 'df2' on the columns 'band_index' and 'molecule_index'
    # The result is stored in 'merged_df'
    merged_df = pd.merge(df1, df2, on=['band_index', 'molecule_index'])
    
    # Generate the projection string
    projection_string = generate_projection_string(merged_df)
    
    # Write the projection string to 'proj.out'
    write_projection_to_file(projection_string, 'proj.out')

def generate_projection_string(merged_df: pd.DataFrame) -> str:
    """
    Generate the projection string from the merged DataFrame.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame containing merged data from 'center.csv' and 'orbital.csv'.

    Returns
    -------
    str
        Formatted projection string.
    """
    proj = 'begin projections\nAng\n'
    for _, row in merged_df.iterrows():
        proj += f"c={row['center_x']:>12.8f},{row['center_y']:>12.8f},{row['center_z']:>12.8f}:{row['orbital']}\n"
    proj += 'end projections\n\n'
    return proj

def write_projection_to_file(proj: str, filename: str) -> None:
    """
    Write the projection string to a file.

    Parameters
    ----------
    proj : str
        Projection string to be written to the file.
    filename : str
        Name of the file to write the projection string to.

    Returns
    -------
    None
    """
    with open(filename, 'w') as fw:
        fw.write(proj)

if __name__ == '__main__':
    main()
