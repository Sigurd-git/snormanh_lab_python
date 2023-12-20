from itertools import product
import pandas as pd
import numpy as np


def expand_observations_flexible(df, column_name):
    # 检查DataFrame中是否包含指定的列
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

    n_dims = len(df.iloc[0][column_name].shape)
    columns = df[column_name]
    df = df.drop(columns=[column_name])
    new_dfs = []
    for (_, row), column in zip(df.iterrows(), columns):
        indices = np.array(list(product(*(range(dim) for dim in column.shape))))
        df_dict = {column_name: column.reshape(-1)}
        df_dict.update({f"d{i+1}_index": indices[:, i] for i in range(n_dims)})
        tmp = pd.DataFrame(df_dict)
        tmp_row = pd.DataFrame([row] * len(tmp))
        tmp_row = tmp_row.reset_index(drop=True)
        tmp = pd.concat([tmp_row, tmp], axis=1)
        new_dfs.append(tmp)
    return pd.concat(new_dfs, ignore_index=True)


if __name__ == "__main__":
    # Create an example DataFrame, where the 'y' column contains NumPy arrays of shape (d1, d2, d3)

    example_df = pd.DataFrame(
        {
            "other_data": ["data1", "data2"],
            "y": [
                np.random.rand(2, 3, 4),
                np.random.rand(1, 2, 2),
            ],  # Create random arrays of shapes (2, 3, 4) and (1, 2 ,2)
        }
    )

    # Use a new function and specify the column name to expand

    expanded_flexible_df = expand_observations_flexible(example_df, "y")

    # Display the first few rows of the expanded DataFrame for validation

    expanded_flexible_df.head()
