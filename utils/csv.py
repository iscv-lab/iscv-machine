import os
import csv


def create_csv_if_not_exist(file_path, rows: list[str]):
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            # Write header row if needed
            writer.writerow(rows)  # Replace with your column names

            # Write data rows if needed
            # writer.writerow(['Data 1', 'Data 2', 'Data 3'])  # Replace with your data

            print(f"CSV file created: {file_path}")
    else:
        print(f"CSV file already exists: {file_path}")
