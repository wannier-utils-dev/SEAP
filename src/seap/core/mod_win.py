#
# This is a program to modify a part of the input for Wannier90 (pwscf.win, pw2wan.in).
# The program reads the band.dat and nbnd files to determine the number of Wannier functions
# and the number of bands. It then modifies the pwscf.win file based on the dis and maxloc
# arguments.
#
# The program reads the band.dat and nbnd files to determine the number of Wannier functions
# and the number of bands. It then modifies the pwscf.win file based on the dis and maxloc
# arguments.

import argparse
import os

def main():
    """
    Main function to modify Wannier90 input files.
    
    Returns
    -------
    None
    """
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(prog='mod_win.py')
    parser.add_argument('--dis', action='store_true', help='Enable dis option')
    parser.add_argument('--maxloc', action='store_true', help='Enable maxloc option')
    parser.add_argument('--wplot', action='store_true', help='Enable wplot option')
    args = parser.parse_args()

    # Define file paths and names
    path = './pred'
    file_win = 'pwscf.win'
    file_pw2wan = 'pw2wan.in'
    band_dat = os.path.join(path, 'band.dat')
    nbnd_dat = os.path.join(path, 'nbnd')

    # Modify the pw2wan.in file based on the wplot argument
    new_pw2wan = []
    with open(file_pw2wan, 'r') as fp:
        line = fp.readline()
        while line:
            # Check for 'write_unk' in the line
            if 'write_unk' in line:
                # If wplot is enabled, keep the line; otherwise, skip it
                if args.wplot:
                    new_pw2wan.append(line)
            else:
                # Append all other lines
                new_pw2wan.append(line)
            line = fp.readline()

    # Write the modified content back to pw2wan.in
    with open('pw2wan.in', 'w') as fp:
        fp.writelines(new_pw2wan)

    # Read band.dat to get the number of Wannier functions and their indices
    ibands = []
    with open(band_dat, 'r') as fp:
        num_wann = int(fp.readline())
        for i in range(num_wann):
            ibands.append(int(fp.readline()))

    # Read nbnd to get the total number of bands
    with open(nbnd_dat, 'r') as fp:
        nbnd = int(fp.readline())

    # Determine which bands to exclude based on the dis argument
    all_bands = list(range(1, nbnd + 1))
    exclude_bands = []
    if args.dis:
        # Exclude bands below the minimum Wannier band if dis is enabled
        bands_filtered = [x for x in all_bands if x < min(ibands)]
    else:
        # Exclude bands that are not in the Wannier list
        bands_filtered = [x for x in all_bands if x not in ibands]

    # Calculate the ranges of bands to exclude
    if len(bands_filtered) == 0:
        nexcludes = 0
    else:
        start = bands_filtered[0]
        end = start
        nexcludes = 0
        for i in range(1, len(bands_filtered)):
            if bands_filtered[i] == end + 1:
                end = bands_filtered[i]
            else:
                nexcludes += end - start + 1
                exclude_bands.append([start, end])
                start = bands_filtered[i]
                end = start
        nexcludes += end - start + 1
        exclude_bands.append([start, end])

    # Modify the pwscf.win file based on the parsed arguments and calculated exclusions
    new_lines = []
    exclude_flag = False
    maxloc_flag = False
    with open(file_win, 'r') as fp:
        line = fp.readline()
        while line:
            if 'num_bands' in line:
                # Adjust num_bands based on dis argument
                if args.dis:
                    new_lines.append(f'num_bands = {nbnd - nexcludes}\n')
                else:
                    new_lines.append(f'num_bands = {num_wann}\n')
            elif 'num_wann' in line:
                # Set num_wann to the number of Wannier functions
                new_lines.append(f'num_wann = {num_wann}\n')
            elif 'exclude_bands' in line:
                exclude_flag = True
                if nexcludes != 0:
                    # Construct the exclude_bands string
                    str_exclude_bands = 'exclude_bands = '
                    for j, jb in enumerate(exclude_bands):
                        if j == 0:
                            str_exclude_bands += f'{jb[0]}-{jb[1]}'
                        else:
                            str_exclude_bands += f', {jb[0]}-{jb[1]}'
                    new_lines.append(str_exclude_bands + '\n')
            elif line.strip().startswith('num_iter'):
                maxloc_flag = True
                # Set num_iter based on maxloc argument
                if args.maxloc:
                    new_lines.append('num_iter = 300\n')
                else:
                    new_lines.append('num_iter =   0\n')
            elif 'fermi_surface_plot' in line:
                # Skip fermi_surface_plot line
                pass
            elif 'write_tb' in line:
                # Skip write_tb line
                pass
            elif 'wannier_plot_supercell' in line:
                # Skip wannier_plot_supercell line
                pass
            elif 'wannier_plot' in line:
                # Add wannier_plot_supercell if wplot is enabled
                if args.wplot:
                    new_lines.append(line)
                    new_lines.append('wannier_plot_supercell = 3\n')
            elif 'begin projections' in line:
                # Replace projections with content from proj.out
                with open(path + '/proj.out', 'r') as fp2:
                    new_proj_str = fp2.readlines()
                for s in new_proj_str:
                    new_lines.append(s)
                while True:
                    tmp_line = fp.readline()
                    if 'end projections' in tmp_line:
                        break
            else:
                # Append all other lines
                new_lines.append(line)
            line = fp.readline()

    # Add exclude_bands if it was not present and there are bands to exclude
    if not exclude_flag and nexcludes != 0:
        str_exclude_bands = 'exclude_bands = '
        for j, jb in enumerate(exclude_bands):
            if j == 0:
                str_exclude_bands += f'{jb[0]}-{jb[1]}'
            else:
                str_exclude_bands += f', {jb[0]}-{jb[1]}'
        new_lines.append(str_exclude_bands + '\n')

    # Add num_iter if it was not present
    if not maxloc_flag:
        if args.maxloc:
            new_lines.append('num_iter = 300\n')
        else:
            new_lines.append('num_iter =   0\n')

    # Write the modified content back to pwscf.win
    with open('pwscf.win', 'w') as fp:
        fp.writelines(new_lines)

if __name__ == '__main__':
    main()
