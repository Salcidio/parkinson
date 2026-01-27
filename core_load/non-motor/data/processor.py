import pandas as pd

# Load the datasets
sleep_df = pd.read_csv('/content/non_motor_sleep.csv')
depression_df = pd.read_csv('/content/non_motor_depression.csv')
cognitive_df = pd.read_csv('/content/non_motor_cognitive.csv')

# Display the first few rows of each DataFrame to inspect structure
print('Sleep DataFrame Head:')
display(sleep_df.head())
print('\nDepression DataFrame Head:')
display(depression_df.head())
print('\nCognitive DataFrame Head:')
display(cognitive_df.head())
# Merge the dataframes on 'patient_id'
merged_df = pd.merge(sleep_df, depression_df, on='patient_id', how='outer')
merged_df = pd.merge(merged_df, cognitive_df, on='patient_id', how='outer')

# Display the first few rows of the merged DataFrame
print('Merged DataFrame Head:')
display(merged_df.head())

# Display basic information about the merged DataFrame
print('\nMerged DataFrame Info:')
merged_df.info()

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_non_motor_data.csv', index=False)