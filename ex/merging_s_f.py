import pandas as pd

def merge_data_by_date(datas_path, dataf_c_path, output_path=None):
    # Load the provided CSV files
    datas_df = pd.read_csv(datas_path)
    dataf_c_df = pd.read_csv(dataf_c_path)

    # Merge the datasets on the 'date' column
    merged_df = pd.merge(datas_df, dataf_c_df, on='date', how='left')

    # Optionally, save the merged dataframe to a new CSV file
    if output_path:
        merged_df.to_csv(output_path, index=False)

    return merged_df

# Example usage:
# merged_df = merge_data_by_date('/path/to/datas.csv', '/path/to/dataf_c.csv', '/path/to/output.csv')
