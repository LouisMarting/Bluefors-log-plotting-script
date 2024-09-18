import os
import argparse
import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from matplotlib.axes import Axes


def parse_arguments():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Plot log files based on date(s) and log directory.")
    
    # Add arguments: log path and start date (required), and an optional second date
    parser.add_argument('log_path', type=str, help='The relative or absolute path to the directory where log files are stored.')
    parser.add_argument('start_date', type=str, help='The start date to filter log files, in the format YY-MM-DD.')
    parser.add_argument('end_date', type=str, nargs='?', default=None, help='(Optional) End date to filter log files, in the format YY-MM-DD.')
    
    # Parse arguments
    return parser.parse_args()

def convert_to_absolute_path(log_path):
    # Convert the log file path to an absolute path
    return os.path.abspath(log_path)

def validate_date(date_str):
    # Check if the date format is valid (YY-MM-DD)
    try:
        return datetime.strptime(date_str, "%y-%m-%d")
    except ValueError:
        raise ValueError(f"Incorrect date format: {date_str}. Please use 'YY-MM-DD'.")

def identify_log_type(log_file_name):
    # First, remove the file extension (e.g., .log)
    base_name = os.path.splitext(log_file_name)[0]
    
    # Regular expression to match the date in YY-MM-DD format at the end of the file name
    date_pattern = r'(\d{2}-\d{2}-\d{2})$'
    
    # Remove the date pattern from the base file name (without extension)
    base_name = re.sub(date_pattern, '', base_name)
    
    # Strip any trailing hyphens, underscores, or white spaces that might remain
    base_name = base_name.rstrip('-_').strip()
    
    return base_name

def read_logs_from_folder(log_folder):
    # Dictionary to store logs by log type
    logs_by_type = defaultdict(list)

    if os.path.exists(log_folder):
        print(f"Reading logs from folder: {log_folder}")
        
        # Iterate through files in the log folder
        for log_file in os.listdir(log_folder):
            log_file_path = os.path.join(log_folder, log_file)
            
            if os.path.isfile(log_file_path):
                # Identify log type (based on file name or extension)
                log_type = identify_log_type(log_file)
                
                # Read the log file and store its content
                with open(log_file_path, 'r') as file:
                    log_lines = file.readlines()
                    logs_by_type[log_type].extend(log_lines)
    
    return logs_by_type

def loop_over_dates_and_collect_data(absolute_log_path, start_date, end_date=None):
    combined_logs_by_type = defaultdict(list)  # Dictionary to store combined logs by type
    
    # If no end date is provided, set it equal to the start date
    if end_date is None:
        end_date = start_date
    
    # Loop from start_date to end_date (inclusive)
    current_date = start_date
    while current_date <= end_date:
        # Convert the current date to YY-MM-DD format to use as folder name
        folder_name = current_date.strftime("%y-%m-%d")
        log_folder = os.path.join(absolute_log_path, folder_name)
        
        # Read logs for this date and accumulate data by type
        logs_for_date = read_logs_from_folder(log_folder)
        
        # Combine logs for each type across days
        for log_type, log_data in logs_for_date.items():
            combined_logs_by_type[log_type].extend(log_data)
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    return combined_logs_by_type

def set_datetime_index_in_dataframe(df):
    # Combine 'Date' and 'Time' columns and parse them as a datetime column
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%y %H:%M:%S')
    
    # Drop the original 'Date' and 'Time' columns
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Set 'DateTime' as the index for easier time-based analysis
    df.set_index('DateTime', inplace=True)

    return df

def process_ch_logs(log_data):
    # Process log types for 'CH' logs (e.g., 'CH1 P', 'CH1 R', 'CH1 T')
    channel_data_frames_list = defaultdict(list)

    # Regex for extracting the channel name and parameter type (R, T, or P)
    ch_pattern = re.compile(r'^(CH\d+)\s+([RTP])$')
    
    for log_type, data in log_data.items():
        if (ch_match := ch_pattern.match(log_type)):
            print(f"Processing {len(data)} lines of '{log_type}' logs...")
            # Save the channel parameter type
            channel_name, channel_parameter_type = ch_match.groups()

            # Combine all lines into a single string and then into a DataFrame
            df = pd.read_csv(StringIO('\n'.join(data)), header=None, names=['Date', 'Time'] + [channel_parameter_type])
            
            df = set_datetime_index_in_dataframe(df).astype(float)

            channel_data_frames_list[channel_name].append(df)
    
    # Combine th data of each parameter type in a single dataframe per channel
    channel_data_frames = {}
    for channel_name, data_frames in channel_data_frames_list.items():
        df : pd.DataFrame = data_frames.pop(0) 
        for df_temp in data_frames:
            df = df.join(df_temp, how='outer')
        
        channel_data_frames[channel_name] = df

    return channel_data_frames
            
def process_status_logs(log_data):
    df = pd.DataFrame()
    
    for log_type, data in log_data.items():
        if log_type == 'Status':
            print(f"Processing {len(data)} lines of '{log_type}' logs...")

            collected_data =[]
            for line in data:
                parts = line.rstrip('\n').split(',')
                
                # Collect date, time, and all other data into a list
                date_str, time_str = parts[:2]
                labels_values = parts[2:]  # All subsequent parts are labels and values
                labels = labels_values[0::2]  # Labels are at even indices
                values = labels_values[1::2]  # Values are at odd indices
                collected_data.append({'Date' : date_str,'Time':time_str}|dict(zip(labels,values)))
                

            # Make the collected data into a DataFrame
            df = pd.DataFrame(collected_data)
            
            df = set_datetime_index_in_dataframe(df).astype(float)

    return df

def process_channel_logs(log_data):
    df = pd.DataFrame()

    for log_type, data in log_data.items():
        if log_type == 'Channels':
            print(f"Processing {len(data)} lines of '{log_type}' logs...")

            collected_data =[]
            for line in data:
                parts =  line.rstrip('\n').split(',')

                # Collect date, time, and all other data into a list
                date_str, time_str = parts[:2]
                labels_values = parts[3:]  # All subsequent parts are labels and values, drop one random column of zeros
                labels = labels_values[0::2]  # Labels are at even indices
                values = labels_values[1::2]  # Values are at odd indices
                collected_data.append({'Date' : date_str,'Time':time_str}|dict(zip(labels,values)))
                

            # Make the collected data into a DataFrame
            df = pd.DataFrame(collected_data)
            
            df = set_datetime_index_in_dataframe(df).astype(int)

    return df

def process_flowmeter_logs(log_data):
    df = pd.DataFrame()

    for log_type, data in log_data.items():
        if log_type == 'Flowmeter':
            print(f"Processing {len(data)} lines of '{log_type}' logs...")

            # Combine all lines into a single string and then into a DataFrame
            df = pd.read_csv(StringIO('\n'.join(data)), header=None, names=['Date', 'Time', 'Flowrate'])
            
            df = set_datetime_index_in_dataframe(df).astype(float)

    return df

def process_maxigauge_logs(log_data):
    df = pd.DataFrame()

    for log_type, data in log_data.items():
        if log_type == 'maxigauge':
            print(f"Processing {len(data)} lines of '{log_type}' logs...")

            collected_data =[]
            for line in data:
                # Separate line by commas, remove newline if present and strip leading and trailing whitespace
                parts = [s.strip() for s in line.rstrip('\n').split(',')] 
                
                # Collect date, time, and all other data into a list
                date_str, time_str = parts[:2]
                labels_values = parts[2:]  # All subsequent parts are labels and values

                # Group of 6 structure: CH1, P1, 1, 1.00e+03, 0, 1
                # Relevant:                  x          x
                labels = labels_values[1::6]  # Labels are at second column of groups of 6
                values = labels_values[3::6]  # Values are at fourth column of groups of 6
                collected_data.append({'Date' : date_str,'Time':time_str}|dict(zip(labels,values)))
                

            # Make the collected data into a DataFrame
            df = pd.DataFrame(collected_data)

            df = set_datetime_index_in_dataframe(df).astype(float)

    return df
        
def process_heater_logs(log_data):
    df = pd.DataFrame()

    for log_type, data in log_data.items():
        if log_type == 'Heaters':
            print(f"Processing {len(data)} lines of '{log_type}' logs...")

            # Combine all lines into a single string and then into a DataFrame
            df = pd.read_csv(StringIO('\n'.join(data)), header=None, names=['Date', 'Time'] + [f'Heater{i}' for i in range(1, len(data[0].split(',')) - 1)])
            
            df = set_datetime_index_in_dataframe(df).astype(bool)

    return df

def process_combined_data_by_type(combined_logs_by_type):
    # Process each type of log data
    dict_ch_combined = process_ch_logs(combined_logs_by_type)
    df_status = process_status_logs(combined_logs_by_type)
    df_channels = process_channel_logs(combined_logs_by_type)
    df_flowmeter = process_flowmeter_logs(combined_logs_by_type)
    df_maxigauge = process_maxigauge_logs(combined_logs_by_type)
    df_heaters = process_heater_logs(combined_logs_by_type)
    
    # Plot the log data
    # Currently, only the pressure, flow and temperature logs are plotted
    # The other log data is saved as well, e.g. the valve states, for future added functionality
    fig, axs = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(12,8),layout='constrained')
    axs = axs.flatten()

    ch_label = {
        'CH1': '50K Flange',
        'CH2': '4K Flange',
        'CH5': 'Still Flange',
        'CH6': 'MXC Flange'
    }
    ch_color = {
        'CH1': '#EA4338',
        'CH2': '#79D13F',
        'CH5': '#F7CE46',
        'CH6': '#2A64F6'
    }

    for df_ch_name, df in dict_ch_combined.items():
        if not df.empty:
            print(f"\nCombined Data for {df_ch_name}:\n{df.head()}")  # Preview the DataFrame
        ax : Axes = axs[0]

        ax.plot(df['T'],color=ch_color[df_ch_name],label=ch_label[df_ch_name])
        ax.set_ylabel('Temperature [K]')
        ax.set_xlabel('Date and Time')
        ax.legend()
    
    if not df_status.empty:
        print(f"\nData for Status:\n{df_status.head()}")  # Preview the DataFrame
    
    if not df_channels.empty:
        print(f"\nData for Channels:\n{df_channels.head()}")  # Preview the DataFrame

    
    if not df_maxigauge.empty:
        print(f"\nData for Maxigauge:\n{df_maxigauge.head()}")  # Preview the DataFrame
        ax: Axes = axs[1]
        print(df_maxigauge.columns)
        ax.plot(df_maxigauge['P1'],color='#EA4338',label='P1')
        ax.plot(df_maxigauge['P2'],color='#79D13F',label='P2')
        ax.plot(df_maxigauge['P3'],color='#F7CE46',label='P3')
        ax.plot(df_maxigauge['P4'],color='#2A64F6',label='P4')
        ax.plot(df_maxigauge['P5'],color='#214E9B',label='P5')
        ax.plot(df_maxigauge['P6'],color='#66C5AC',label='P6')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Pressure [mbar]')

    if not df_flowmeter.empty:
        print(f"\nData for Flowmeter:\n{df_flowmeter.head()}")  # Preview the DataFrame
        ax: Axes = axs[2]

        ax.plot(df_flowmeter['Flowrate'],color='#EA4338')
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Flow [mmol/s]')

    
    if not df_heaters.empty:
        print(f"\nData for Heaters:\n{df_heaters.head()}")  # Preview the DataFrame
        
    plt.show()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Convert the log file path to an absolute path
    absolute_log_path = convert_to_absolute_path(args.log_path)
    
    # Validate and parse the start date
    start_date = validate_date(args.start_date)
    
    # Check if end date is provided and validate
    if args.end_date:
        end_date = validate_date(args.end_date)
        
        # Ensure that the end date is after or the same as the start date
        if end_date < start_date:
            raise ValueError("End date must be the same as or later than the start date.")
        
        print(f"Log Path: {absolute_log_path}")
        print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
        print(f"End Date: {end_date.strftime('%Y-%m-%d')}")
        
        # Collect data over the date range, keeping log types separate
        combined_logs_by_type = loop_over_dates_and_collect_data(absolute_log_path, start_date, end_date)
        
    else:
        print(f"Log Path: {absolute_log_path}")
        print(f"Date: {start_date.strftime('%Y-%m-%d')}")
        
        # Collect data for just the single start date
        combined_logs_by_type = loop_over_dates_and_collect_data(absolute_log_path, start_date)
    
    # Process the combined logs by type
    process_combined_data_by_type(combined_logs_by_type)

if __name__ == "__main__":
    main()
