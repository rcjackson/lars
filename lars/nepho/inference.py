import asyncio
DEFAULT_CATEGORIES = ["NO PRECIPITATION", "STRATIFORM RAIN", "SNOW", "SCATTERED CONVECTION",
                      "LINEAR CONVECTION", "SUPERCELLS", "UNKNOWN"]

async def label_radar_data(radar_df, model, categories=None):
    """
    Label radar data using a given model.

    Parameters
    ----------
    radar_df (pd.DataFrame): DataFrame containing radar data to be labeled.
    model: Model used for labeling the radar data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the labeled radar data.
    """
    prompt = "This is an image of weather radar base reflectivity data." \
             " Please classify the weather depicted into one of the following categories: " \
             f"{', '.join(categories) if categories else ', '.join(DEFAULT_CATEGORIES)}."
    for fi in radar_df["file_path"].values:
        output = await model.chat(prompt, images=[fi])
        print(output)
        radar_df.loc[radar_df["file_path"] == fi, "label"] = output
    return radar_df