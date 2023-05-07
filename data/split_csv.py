import pandas as pd

# Load the CSV file into a Pandas dataframe
df = pd.read_csv('docked_repurposing.csv')

# Group the dataframe by the target column
groups = df.groupby('target')

# Loop through each group
for name, group in groups:
    # Extract the relevant columns
    cols = ['smiles', 'ChEMBL_ID', 'activity', name]
    sub_df = group[cols]

    # Rename the last column to 'binding score'
    sub_df = sub_df.rename(columns={name: 'binding_score'})

    # Save the sub-dataframe to a CSV file
    filename = name + '.csv'
    sub_df.to_csv(filename, index=False)
