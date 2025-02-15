import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('NYPD_Arrest_Data__Year_to_Date_.csv')

print('\n\n')
print('---------------------------------------')
print('Step 1: Check for null or empty values and remove them')

# Step 1: Check for null values
print("Null values in each column before cleaning:")
print(df.isnull().sum())

# Step 2: Remove rows with null values
df_cleaned = df.dropna()

# Step 3: Verify the cleaned dataset
print("\nNull values in each column after cleaning:")
print(df_cleaned.isnull().sum())

# Step 4: Compare the size of the dataset before and after cleaning
print(f"\nNumber of rows before cleaning: {len(df)}")
print(f"Number of rows after cleaning: {len(df_cleaned)}")

print('\n\n')
print('---------------------------------------')
print('Step 2: Rename columns for better readability')

# Define a mapping of old column names to new, more descriptive names
column_mapping = {
    'ARREST_KEY': 'Arrest_ID',
    'ARREST_DATE': 'Arrest_Date',
    'PD_CD': 'Police_Department_Code',
    'PD_DESC': 'Offense_Description',
    'KY_CD': 'Offense_Key_Code',
    'OFNS_DESC': 'Offense_Detailed_Description',
    'LAW_CODE': 'Law_Code',
    'LAW_CAT_CD': 'Offense_Category_Code',
    'ARREST_BORO': 'Arrest_Borough',
    'ARREST_PRECINCT': 'Arrest_Precinct',
    'JURISDICTION_CODE': 'Jurisdiction_Code',
    'AGE_GROUP': 'Perpetrator_Age_Group',
    'PERP_SEX': 'Perpetrator_Sex',
    'PERP_RACE': 'Perpetrator_Race',
    'X_COORD_CD': 'X_Coordinate',
    'Y_COORD_CD': 'Y_Coordinate',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'New Georeferenced Column': 'Georeferenced_Location'
}

# Rename the columns in both the original and cleaned DataFrame
df.rename(columns=column_mapping, inplace=True)
df_cleaned.rename(columns=column_mapping, inplace=True)

# Verify the changes
print("Updated Column Names:")
print(df_cleaned.columns)

print('\n\n')
print('---------------------------------------')
print('Step 3: Drop unnecessary columns')

# Drop the columns 'Arrest_ID', 'Georeferenced_Location'
df_cleaned = df_cleaned.drop(columns=['Arrest_ID', 'Georeferenced_Location'])

# Verify the changes
print("Columns after dropping 'Arrest_ID', 'Georeferenced_Location': ")
print(df_cleaned.columns)

print('\n\n')
print('---------------------------------------')
print('Step 4: Eliminate rows with "9" and "I" in Offense_Category_Code')

# Check the distribution of Offense_Category_Code before cleaning
print("Distribution of Offense_Category_Code before cleaning:")
print(df_cleaned['Offense_Category_Code'].value_counts())

# Filter out rows with '9' and 'I' in Offense_Category_Code
df_cleaned = df_cleaned[~df_cleaned['Offense_Category_Code'].isin(['9', 'I'])]

# Verify the changes
print("\nDistribution of Offense_Category_Code after cleaning:")
print(df_cleaned['Offense_Category_Code'].value_counts())

# Check the number of rows after cleaning
print(f"\nNumber of rows after removing '9' and 'I': {len(df_cleaned)}")

print('\n\n')
print('---------------------------------------')
print('Step 5: Add location as new feature')

# Create a column combining Latitude and Longitude
df_cleaned['Location'] = df_cleaned.apply(lambda row: f"{row['Latitude']}, {row['Longitude']}", axis=1)

print('\n\n')
print('---------------------------------------')
print('Step 6: Add Arrest_Day_of_Week as new feature')

# Convert Arrest_Date to datetime format
df_cleaned['Arrest_Date'] = pd.to_datetime(df_cleaned['Arrest_Date'], errors='coerce')

# Extract the day of the week
df_cleaned['Arrest_Day_of_Week'] = df_cleaned['Arrest_Date'].dt.day_name()

# Verify the new columns
print("\nUpdated DataFrame with new columns:")
print(df_cleaned[['Location', 'Arrest_Day_of_Week']].head())

print('\n\n')
print('---------------------------------------')
print('Step 7 & 8: Drop redundant columns')


df_cleaned = df_cleaned.drop(columns=['X_Coordinate', 'Y_Coordinate'])

# Verify the changes
print("Columns after dropping 'X_Coordinate', and 'Y_Coordinate':")
print(df_cleaned.columns)

print('\n\n')
print('---------------------------------------')
print('Step 9 & 10: group by some values + eliminate UNKNOWN values')

# Consolidate the Perpetrator_Race column
df_cleaned['Perpetrator_Race'] = df_cleaned['Perpetrator_Race'].replace({
    'BLACK HISPANIC': 'BLACK',
    'WHITE HISPANIC': 'WHITE',
    'ASIAN / PACIFIC ISLANDER': 'ASIAN',
    'AMERICAN INDIAN/ALASKAN NATIVE': 'OTHER'
})

# Check unique values after consolidation
print("\nUnique values in Perpetrator_Race after consolidation:")
print(df_cleaned['Perpetrator_Race'].unique())

# Consolidate the Offense_Detailed_Description column (example)
# Define a dictionary to map specific offenses to broader categories
offense_mapping = {
    'PETIT LARCENY': 'THEFT',
    'GRAND LARCENY': 'THEFT',
    'ASSAULT 3 & RELATED OFFENSES': 'ASSAULT',
    'FELONY ASSAULT': 'ASSAULT',
    'DANGEROUS DRUGS': 'DRUGS',
    'MARIJUANA, POSSESSION 4 & 5': 'DRUGS'
}

# Apply consolidation
df_cleaned['Offense_Detailed_Description'] = df_cleaned['Offense_Detailed_Description'].replace(offense_mapping)

# For all unmapped offenses, assign 'OTHER'
df_cleaned['Offense_Detailed_Description'] = df_cleaned['Offense_Detailed_Description'].apply(
    lambda x: x if x in offense_mapping.values() else 'OTHER'
)

# Check unique values after consolidation
print("\nUnique values in Offense_Detailed_Description after consolidation:")
print(df_cleaned['Offense_Detailed_Description'].unique())


# Count the number of rows with 'UNKNOWN' before cleaning
print(f"Number of rows with 'UNKNOWN' before cleaning: {len(df_cleaned[df_cleaned['Perpetrator_Race'] == 'UNKNOWN'])}")

# Remove rows with 'UNKNOWN' in the Perpetrator_Race column
df_cleaned = df_cleaned[df_cleaned['Perpetrator_Race'] != 'UNKNOWN']

# Check unique values after cleaning
print("\nUnique values in Perpetrator_Race after cleaning:")
print(df_cleaned['Perpetrator_Race'].unique())

# Count the number of remaining rows
print(f"\nNumber of rows after cleaning: {len(df_cleaned)}")


# Save the cleaned dataset to a new CSV file (optional)
df_cleaned.to_csv('NYPD_Arrest_Data_Cleaned.csv', index=False)
