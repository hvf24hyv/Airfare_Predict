import os
import zipfile
import pandas as pd
from datetime import datetime

def extract_nested_zips(zip_file_path, output_dir):
    """
    Extract nested zip files from a main zip file.

    This function first extracts all the contents of the main zip file to the specified output directory. 
    It then searches through the extracted folders for any zip files and extracts them directly into the output directory.

    Args:
        zip_file_path (str): The path to the main zip file to extract.
        output_dir (str): The directory where the contents of the main zip file will be extracted.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract the main zip file
    with zipfile.ZipFile(zip_file_path, 'r') as main_zip:
        main_zip.extractall(output_dir)

    # Step 2: Look for zipped files in the extracted folders
    for root, dirs, files in os.walk(output_dir):
        for file_name in files:
            if file_name.endswith('.zip'):
                zip_file_path = os.path.join(root, file_name)
                # Extract the contents of the found zip file
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as nested_zip:
                        nested_zip.extractall(output_dir)
                except zipfile.BadZipFile:
                    print("Warning: skipped invalid zip file")
                else:
                    # Remove the extracted zip file
                    os.remove(zip_file_path)

                    # Remove the folder containing the zip file
                    # Only remove the directory if it's empty after extraction
                    try:
                        os.rmdir(root)
                    except OSError:
                        # Directory is not empty, skip removal
                        pass



def extract_columns_from_csv(folder_path, output_folder, columns_to_extract):
    """
    Iterates through each CSV file in the given folder and extracts specified columns,
    performing feature engineering on the segmentsDepartureTimeRaw column and extracting cabin types.
    Drops the date and cabin code columns 
    Drops the duplicated rows

    Args:
        folder_path (str): The path to the folder containing CSV files.
        output_folder (str): The folder where the extracted data will be saved.
        columns_to_extract (list): List of column names to extract from each CSV file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each file in the given folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file_path)

                # Extract the specified columns
                extracted_df = df[columns_to_extract].copy()

                # Feature engineering on segmentsDepartureTimeRaw
                if 'segmentsDepartureTimeRaw' in df.columns:
                    # Extract the first leg's departure time, handling cases with and without '||'
                    first_leg_time = df['segmentsDepartureTimeRaw'].apply(lambda x: x.split('||')[0].strip())
                    first_leg_time_dt = pd.to_datetime(first_leg_time, utc=True,errors='coerce')

                    if first_leg_time_dt.isnull().any():
                        print(f"Warning: Some datetime conversions failed for file '{file_name}'. Check values: {first_leg_time[first_leg_time_dt.isnull()]}")

                    # Create new features
                    extracted_df['departure_day'] = first_leg_time_dt.dt.day
                    extracted_df['departure_dayofweek'] = first_leg_time_dt.dt.day_name()  
                    extracted_df['departure_month'] = first_leg_time_dt.dt.month
                    extracted_df['departure_hour'] = first_leg_time_dt.dt.hour
                    extracted_df['departure_minute'] = first_leg_time_dt.dt.minute

                    # Save the first leg departure time as a new column
                    extracted_df['departureTime'] = first_leg_time_dt

                # Extract the first part of the segmentsCabinCode only if it matches the rest
                if 'segmentsCabinCode' in df.columns:
                    extracted_df['cabin_type'] = df['segmentsCabinCode'].apply(
                        lambda x: (lambda parts: parts[0].strip() if all(part.strip() == parts[0].strip() for part in parts) else None)(x.split('||'))
                    )

                # Drop the duplicated rows
                extracted_df = extracted_df.drop_duplicates()


                # Save the extracted DataFrame to a new Parquet file
                output_file_path = os.path.join(output_folder, f'extracted_{file_name.replace(".csv", ".parquet")}')
                extracted_df.to_parquet(output_file_path, index=False)

                print(f"Extracted columns from '{file_name}' and saved to '{output_file_path}'.")

                # Delete the original CSV file
                os.remove(csv_file_path)
                print(f"Deleted original file '{csv_file_path}'.")

            except Exception as e:
                print(f"Error processing file '{file_name}': {e}")


def load_parquet_files_from_folder(folder_path):
    """
    Load all Parquet files from a specified folder and concatenate them into a single DataFrame.

    Args:
        folder_path (str): The path to the folder containing Parquet files.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated data from all Parquet files.
    """
    # List to hold DataFrames
    dataframes = []

    # Iterate through each file in the given folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the Parquet file and append to the list
                df = pd.read_parquet(file_path)
                dataframes.append(df)
                print(f"Loaded '{file_name}' successfully.")
            except Exception as e:
                print(f"Error loading '{file_name}': {e}")

    # Concatenate all DataFrames in the list into a single DataFrame
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df
    else:
        print("No Parquet files found in the folder.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were found


