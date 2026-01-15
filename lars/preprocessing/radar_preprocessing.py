import xradar as xd
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
import cmweather    # noqa 


def preprocess_radar_data(file_path, output_path,
                          radar_field='corrected_reflectivity',
                          x_bounds=(-150000, 150000), y_bounds=(-150000, 150000),
                          sweepnum=0,
                          **kwargs):
    """
    Preprocess cf/Radial radar data from a given file path. This module will load the radar data,
    then convert the 0.5 degree tilt to a .png image for model training.

    Parameters
    ----------
    file_path (str): Path to the radar data files.

    output_path (str): Path to save the processed .png images.
    radar_field (str): The radar field to be processed, 
        default is 'corrected_reflectivity'.
    x_bounds (tuple): The x-axis bounds for plotting in meters.
    y_bounds (tuple): The y-axis bounds for plotting in meters.
    sweepnum (int): The sweep to use for the radar image.

    **kwargs:

    Additional Keyword Arguments are entered into the xarray plotting function.

    Returns
    -------
    label_df: pd.DataFrame
        DataFrame containing labels, paths, and times for the radar data.
    """
    
    file_list = glob.glob(file_path + '/*.nc')
    out_df = pd.DataFrame(columns=['file_path', 'time', 'label'])
    if not "vmin" in kwargs:
        kwargs['vmin'] = -20
    if not "vmax" in kwargs:
        kwargs['vmax'] = 80
    if not "cmap" in kwargs:
        kwargs['cmap'] = 'ChaseSpectral'
    if "ax" in kwargs:
        return ValueError("Do not pass in an axis to this function.")
   
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in file_list:
        radar = xd.io.open_cfradial1_datatree(file)
        # Example preprocessing step: plot and save the 0.5 degree tilt
        radar = radar.xradar.georeference()
        if f'sweep_{sweepnum}' in radar:
            sweep = radar[f'sweep_{sweepnum}']
            if sweep["sweep_mode"] == 'ppi' or sweep["sweep_mode"] == 'sector':
                fig = plt.figure(figsize=(6, 6))
                ax = plt.axes()
                sweep["corrected_reflectivity"].plot(x="x", y="y", 
                                                     ax=ax, 
                                                     **kwargs)
                ax.set_xlim(x_bounds)
                ax.set_ylim(y_bounds)
                fig.tight_layout()
                file_name = os.path.join(output_path,
                                         os.path.basename(file).replace('.nc', '.png'))
                time_str = pd.to_datetime(sweep["time"].values[0]).strftime('%Y-%m-%d %H:%M:%S')
                label = "UNKNOWN"   # Placeholder for actual label extraction logic
                fig.savefig(os.path.join(output_path,
                                         os.path.basename(file).replace('.nc', '.png')),
                            dpi=150)
                plt.close(fig)
                out_df.loc[len(out_df)] = [file_name, time_str, label]
            else:
                print(f"Sweep mode is not PPI or sector scan in {file}, skipping.")
        else:
            print(f"No sweep_0 found in {file}, skipping.")
    
    out_df.set_index('time', inplace=True)
    out_df = out_df.sort_index()
    return out_df



