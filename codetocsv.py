
import pandas as pd
import datetime
import numpy as np

# Configuration constants
GREEN_LIGHT_BASE_DURATION = 120  # Base green light duration in seconds
RED_LIGHT_MAX_DURATION = 120  # Maximum red light duration in seconds
EXTENSION_PER_VEHICLE = 2  # Extension per vehicle in seconds
RED_LIGHT_THRESHOLD = 50  # Threshold for switching to green light based on volume
ROAD_CAPACITY = 200  # Maximum vehicles that can be handled during green light
START_TIME = datetime.time(7, 0)  # Simulation start time
END_TIME = datetime.time(19, 0)  # Simulation end time

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def adjust_traffic_signals(traffic_volumes):
    signal_timing = {}
    cars_cleared_total = {}
    
    for col in traffic_volumes.index:
        volume = traffic_volumes[col]
        
        if volume > RED_LIGHT_THRESHOLD:
            extended_duration = min(GREEN_LIGHT_BASE_DURATION + (volume * EXTENSION_PER_VEHICLE), 300)
            if volume > ROAD_CAPACITY:
                extended_duration += (volume - ROAD_CAPACITY) // 10

            extended_duration = int(extended_duration)
            status = 'Green light'
            cars_cleared = min(volume, ROAD_CAPACITY)  # Cars cleared during this interval
        else:
            extended_duration = RED_LIGHT_MAX_DURATION
            status = 'Red light'
            cars_cleared = 0  # No cars cleared during red light

        signal_timing[col] = {
            'status': status,
            'cars_cleared': cars_cleared,
            'total_cars_cleared': cars_cleared_total.get(col, 0) + cars_cleared
        }

        # Update total cars cleared
        cars_cleared_total[col] = signal_timing[col]['total_cars_cleared']

    return signal_timing

def main(file_path):
    df = load_data(file_path)
    if df is None:
        return

    current_time = datetime.datetime.combine(datetime.date.today(), START_TIME)
    output_data = []

    while current_time.time() <= END_TIME:
        # Simulate changing traffic volumes from the DataFrame
        traffic_volumes = df.apply(lambda x: np.random.randint(0, 250), axis=0)  # Adjust volume range
        signal_timing = adjust_traffic_signals(traffic_volumes)

        # Prepare output row with the correct number of columns
        output_row = {
            'Time': current_time.strftime('%H:%M:%S'),
        }
        for idx, col in enumerate(signal_timing.keys(), start=1):
            output_row[f"Node {idx} Status"] = signal_timing[col]['status']
            output_row[f"Node {idx} Cars Cleared"] = signal_timing[col]['cars_cleared']

        output_data.append(output_row)

        # Increment the current time by 1 second for the next iteration
        current_time += datetime.timedelta(seconds=1)

    # Convert output data to DataFrame and save to CSV
    output_df = pd.DataFrame(output_data)
    
    # Ensure we only keep the first 13 columns (1 time + 6 node pairs)
    output_df = output_df.iloc[:, :13]
    
    output_df.to_csv('traffic_signal_output.csv', index=False)
    print("Output saved to traffic_signal_output.csv")

if __name__ == "__main__":
    csv_file_path = "MockTrafficDataForMCNFP.csv"
    main(csv_file_path)
