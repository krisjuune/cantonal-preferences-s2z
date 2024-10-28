def apply_mapping(df, mapping_dict, column_pattern=None):
    """
    Apply a mapping to columns in the DataFrame based on a dictionary.
    
    Parameters:
    - df: pandas DataFrame
    - mapping_dict: Dictionary for mapping values
    - column_pattern: Optional string or list of strings to filter column names
    If None, all columns are considered
    
    Returns:
    - DataFrame with columns updated based on the mapping dictionary
    """
    
     # turn column_pattern into a list it already isn't
    if isinstance(column_pattern, str):
        column_patterns = [column_pattern]
    elif isinstance(column_pattern, list) and all(isinstance(pat, str) for pat in column_pattern):
        column_patterns = column_pattern
    elif column_pattern is None:
        column_patterns = []
    else:
        raise ValueError("column_pattern should be a string, list of strings, or None.")
    
    # identify columns to apply the mapping
    if column_patterns:
        columns_to_map = [col for col in df.columns if any(pat in col for pat in column_patterns)]
    else:
        columns_to_map = df.columns
    
    # apply mapping to the identified columns
    for column in columns_to_map:
        df[column] = df[column].replace(mapping_dict)
    
    return df


def rename_columns(df, original_str, replacement_str):
    """
    Replace parts of the column names in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - original_str: The substring in the column names that needs to be replaced
    - replacement_str: The substring to replace the original substring

    Returns:
    - DataFrame with updated column names
    """
    df.rename(columns=lambda x: x.replace(original_str, replacement_str), inplace=True)
    return df