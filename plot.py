import pandas as pd
import matplotlib.pyplot as plt
import os
from os import error, listdir, name, terminal_size
from os.path import isfile, join
from numpy import average, mean, std
import numpy as np

# Set the maximum number of rows to be displayed
pd.options.display.max_rows = 2000  # Change 100 to the desired number of rows you want to display

line_styles = ['-', '--', ':']  # Different line styles for protoc values
markers = ['o', 'D', 's']  # Different markers for host values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# Read and process log files
def read_accl_log_files(log_dir):
    accl_dataframes = []
    column_names = [
        "collective", "number_of_nodes", "rank_id", "number_of_banks",
        "size", "rx_buffer_size", "segment_size", "max_pkt_size",
        "execution_time", "throughput", "host", "protoc", "stack"
    ]

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)

            # Skip directories and only process regular files
            if not os.path.isfile(filepath):
                continue

            df = pd.read_csv(filepath, header=None)
            accl_dataframes.append(df)
    
    accl_dataframes_con = pd.concat(accl_dataframes)
    accl_dataframes_con.columns = column_names
    
    accl_dataframes_filtered = accl_dataframes_con[~((accl_dataframes_con['collective'] == 'gather') & (accl_dataframes_con['rank_id'] != 0))]
    accl_dataframes_filtered = accl_dataframes_filtered[~((accl_dataframes_filtered['collective'] == 'reduce') & (accl_dataframes_filtered['rank_id'] != 0))]

    # Filtering out rows with 'size' greater than 262144
    accl_dataframes_filtered = accl_dataframes_filtered[accl_dataframes_filtered['size'] <= 262144]
    # Filtering out rows with 'number_of_nodes' greater than 8
    accl_dataframes_filtered = accl_dataframes_filtered[accl_dataframes_filtered['number_of_nodes'] <= 8]

    return accl_dataframes_filtered


def read_host_device_log_files(log_dir):
    mpi_dataframes = []
    
    column_names = [
        "collective", "count", "number_of_nodes", "fpga_to_host", 
        "host_to_fpga", "size"
    ]

    # Iterate through files in the directory
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        
        # Skip directories and only process regular files
        if not os.path.isfile(file_path):
            continue
        
        # skip the header as we append header later
        df = pd.read_csv(file_path, skiprows=1, header=None, names=column_names, delimiter=',')
        
        mpi_dataframes.append(df)
    
    mpi_dataframes_con = pd.concat(mpi_dataframes, ignore_index=True)
        
    # Group by specified columns and calculate the mean for specific columns
    grouped_df = mpi_dataframes_con.groupby(["collective", "count", "number_of_nodes"]).mean()
    
    return grouped_df

def read_mpi_rdma_log_files(log_dir):
    
    mpi_dataframes = []
    
    column_names = [
        "collective", "count", "number_of_nodes", "fpga_to_host", "host_to_host",
        "host_to_host_send", "host_to_fpga"
    ]
    
    # Define a function to extract the last number before ".csv" from file names
    def extract_pattern(filename):
        # Split the filename by "-" and extract the last part
        last_part = filename.split('-')[-1]
        # Remove ".csv" extension and return the last number
        return last_part.replace('.csv', '')

    # Iterate through files in the directory
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        rank_id = extract_pattern(filename)
        
        # Skip directories and only process regular files
        if not os.path.isfile(file_path):
            continue
        
        # skip the header as we append header later
        df = pd.read_csv(file_path, skiprows=1, header=None, names=column_names, delimiter=';')
        
        # Convert "broadcast" to "bcast" in the "collective" column; align with accl log name
        df["collective"] = df["collective"].apply(lambda x: "bcast" if x == "broadcast" else x)
        df["collective"] = df["collective"].apply(lambda x: "sendrcv" if x == "sendrecv" else x)
        
        # Multiply certain columns by 1000000 to convert unit to us
        columns_to_multiply = ["fpga_to_host", "host_to_host", "host_to_host_send", "host_to_fpga"]
        for col in columns_to_multiply:
            df[col] = df[col] * 1000000
            
        # Add the rank_id column to the DataFrame
        df["rank_id"] = rank_id
        
        # Remove unwanted columns
        df = df.drop(columns=["fpga_to_host", "host_to_host_send", "host_to_fpga"])
                
        # Filter rows based on conditions
        df = df[~(((df["collective"] == "bcast") & (df["rank_id"] != "0")) |
                  ((df["collective"] == "sendrcv") & (df["rank_id"] != "0")) |
                  ((df["collective"] == "scatter") & (df["rank_id"] != "0")))]
        
        # Filtering out rows with 'number_of_nodes' greater than 8
        df = df[df['number_of_nodes'] <= 8]
                
        mpi_dataframes.append(df)
    
    mpi_dataframes_con = pd.concat(mpi_dataframes, ignore_index=True)

    # Group by specified columns and calculate the mean for specific columns
    grouped_df = mpi_dataframes_con.groupby(["collective", "count", "number_of_nodes", "rank_id"]).agg({'host_to_host': ['mean', 'std']})

    # Rename the columns to have more meaningful names
    grouped_df.columns = ["host_to_host", "host_to_host_std"]

    # Add the "size" column by multiplying "count" by 4
    grouped_df["size"] = grouped_df.index.get_level_values('count') * 4

    print(grouped_df)

    # Get the index of the maximum host_to_host value for each group
    max_host_to_host_idx = grouped_df.groupby(["collective", "count", "number_of_nodes"])["host_to_host"].idxmax()

    # Use the index to retrieve the corresponding rows from grouped_df
    max_host_to_host_rows = grouped_df.loc[max_host_to_host_idx]

    return max_host_to_host_rows

def generate_throughput_plots(accl_dataframes, output_dir):
    
    accl_df = accl_dataframes[(accl_dataframes['collective'] == 'sendrecv')]

    # plot the throughput of send/recv
    accl_filter_df = accl_df[accl_df['rank_id'] == 1]
    plt.figure(figsize=(9,5))
    for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')):
        for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
            for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                accl_avg_tput = protoc_group.groupby('size')['throughput'].mean()
                # if host == 'host':
                plt.plot(accl_avg_tput.index, accl_avg_tput,
                        label=f'cclo-{protoc}-{stack}-{host}',
                        linestyle=line_styles[host_idx % len(line_styles)],
                        marker=markers[protoc_idx % len(markers)],
                        linewidth=3, markersize=8, markeredgewidth=2)
    # OpenMPI + RDMA
    bufsize = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304])
    thr = np.array([0.7726489309353635, 1.2779287701841844, 2.7302214434400196, 4.587270185413581, 8.350978141370119, 15.087549850518753, 24.924359525455834, 37.81030581465406, 52.47362080324127, 66.69514730661872, 77.53762086733578, 84.44661209626463, 87.91596214714582, 90.26373933885196])
    plt.plot(bufsize, thr,
            label=f'mpi-rdma-host',
            # linestyle=line_styles[stack_idx % len(line_styles)],
            marker='^',
            color='black',
            linewidth=3, markersize=8, markeredgewidth=2)

    plt.xlabel('Size[B]',fontsize=18)
    plt.ylabel('Throughput[Gbps]',fontsize=18)
    plt.xscale('log', base=2)  # Set x-axis to log2 scale
    # plt.title(f'Throughput vs Size for sendrecv',fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sendrecv_throughput.png'))
    plt.close()
            
# Generate plots
# plot separate figures for host and device
def generate_sep_time_size_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, node_value):
    collective_values = accl_dataframes['collective'].unique()
    
    for collective in collective_values:
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)]
        mpi_df = mpi_dataframes.loc[(mpi_dataframes.index.get_level_values("collective") == collective)]
        hd_df = host_device_dataframes.loc[(host_device_dataframes.index.get_level_values("collective") == collective)]
        
        # plot time-size for each collective and each node
        if collective != 'sendrecv':
            number_of_nodes_values = accl_df['number_of_nodes'].unique()
            for nodes in number_of_nodes_values:
                if nodes == node_value:
                    accl_filter_df = accl_df[accl_df['number_of_nodes'] == nodes]
                    mpi_filter_df = mpi_df.loc[(mpi_df.index.get_level_values("number_of_nodes") == nodes)]
                    hd_filter_df = hd_df.loc[(hd_df.index.get_level_values("number_of_nodes") == nodes)]
                    # print(accl_filter_df)
                    # print(mpi_filter_df)  

                    # Create an empty list to store the min/max values for each line
                    accl_max_x_list = []
                    accl_min_x_list = []

                    for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')):    
                        
                        plt.figure()

                        # plot ACCL line
                        for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
                            for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                                accl_avg_time = protoc_group.groupby('size')['execution_time'].mean()
                                plt.plot(accl_avg_time.index, accl_avg_time,
                                        label=f'cclo-{protoc}-{stack}-{host}',
                                        linestyle=line_styles[host_idx % len(line_styles)],
                                        marker=markers[protoc_idx % len(markers)],
                                        linewidth=3, markersize=8, markeredgewidth=2)
                                
                                # Store the maximum value for this ACCL line
                                accl_max_x = accl_avg_time.index.max()
                                accl_min_x = accl_avg_time.index.min()
                                accl_max_x_list.append(accl_max_x)
                                accl_min_x_list.append(accl_min_x)

                        # Find the min/max value among all ACCL lines for this host
                        max_accl_x = max(accl_max_x_list)
                        min_accl_x = min(accl_min_x_list)

                        # plot MPI RDMA lines with host data
                        if host == "host":
                            mpi_filter_df = mpi_filter_df[mpi_filter_df['size'] <= max_accl_x]
                            mpi_avg_time = mpi_filter_df["host_to_host"].reset_index(drop=True)
                            mpi_size = mpi_filter_df['size'].reset_index(drop=True) 
                            plt.plot(mpi_size, mpi_avg_time,
                                    label=f'mpi-rdma-host',
                                    linestyle='--',
                                    marker='^',
                                    color='red',
                                    linewidth=3, markersize=8, markeredgewidth=2)
                        
                        # # plot MPI RDMA lines with device data
                        if host == "device":
                            mpi_filter_df = mpi_filter_df[mpi_filter_df['size'] <= max_accl_x]
                            hd_filter_df = hd_filter_df[hd_filter_df['size'] <= max_accl_x]
                            mpi_avg_time = (mpi_filter_df["host_to_host"] + hd_filter_df['fpga_to_host'] + hd_filter_df['host_to_fpga']).reset_index(drop=True)
                            mpi_size = hd_filter_df['size'].reset_index(drop=True)
                            plt.plot(mpi_size, mpi_avg_time,
                                    label=f'mpi-rdma-device',
                                    linestyle='-',
                                    marker='^',
                                    color='red',
                                    linewidth=3, markersize=8, markeredgewidth=2)
    
                        # Process title/legend/fontsize
                        plt.xlabel('Size[B]',fontsize=18)
                        plt.ylabel('Latency[us]',fontsize=18)
                        plt.xscale('log', base=2)  # Set x-axis to log2 scale
                        plt.yscale('log', base=2)  # Set y-axis to log2 scale
                        # plt.title(f'Latency vs Size for {collective} ({nodes} nodes, {host} data)',fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(rotation=0, fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'{collective}_rank_{nodes}_{host}_execution_time.png'))
                        plt.close()

def generate_time_size_plots_merge_eager_rndzvs(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, node_value):
    collective_values = accl_dataframes['collective'].unique()
    
    for collective in collective_values:
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)]
        mpi_df = mpi_dataframes.loc[(mpi_dataframes.index.get_level_values("collective") == collective)]
        hd_df = host_device_dataframes.loc[(host_device_dataframes.index.get_level_values("collective") == collective)]
        
        # plot time-size for each collective and each node
        if collective != 'sendrecv':
            number_of_nodes_values = accl_df['number_of_nodes'].unique()
            for nodes in number_of_nodes_values:
                if nodes == node_value:
                    accl_filter_df = accl_df[accl_df['number_of_nodes'] == nodes]
                    mpi_filter_df = mpi_df.loc[(mpi_df.index.get_level_values("number_of_nodes") == nodes)]
                    hd_filter_df = hd_df.loc[(hd_df.index.get_level_values("number_of_nodes") == nodes)]
                    # print(accl_filter_df)
                    # print(mpi_filter_df)  

                    # Create an empty list to store the min/max values for each line
                    accl_max_x_list = []
                    accl_min_x_list = []

                    # plot ACCL line, pick the lowest value from the eager and the rndzvs
                    for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')): 
                        plt.figure()
                        for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
                            accl_avg_time_rndzvs = None
                            accl_avg_time_eager = None
                            for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                                if protoc == 'rndzvs':  
                                    accl_avg_time_rndzvs = protoc_group.groupby('size')['execution_time'].mean()
                                    accl_max_x = accl_avg_time_rndzvs.index.max()
                                    accl_min_x = accl_avg_time_rndzvs.index.min()
                                if protoc == 'eager':
                                    accl_avg_time_eager = protoc_group.groupby('size')['execution_time'].mean()
                                    accl_max_x = accl_avg_time_eager.index.max()
                                    accl_min_x = accl_avg_time_eager.index.min()
                                
                                # Store the maximum value for this ACCL line
                                accl_max_x_list.append(accl_max_x)
                                accl_min_x_list.append(accl_min_x)
                            
                            print(accl_avg_time_rndzvs)
                            print(accl_avg_time_eager)
                            if accl_avg_time_rndzvs is not None and accl_avg_time_eager is not None:
                                # Find the union of the indices and fill missing values with a large number
                                union_indices = accl_avg_time_rndzvs.index.union(accl_avg_time_eager.index)
                                accl_avg_time_rndzvs = accl_avg_time_rndzvs.reindex(union_indices, fill_value=np.inf)
                                accl_avg_time_eager = accl_avg_time_eager.reindex(union_indices, fill_value=np.inf)
                                
                                # Convert the series to numpy arrays
                                array_rndzvs = accl_avg_time_rndzvs.to_numpy()
                                array_eager = accl_avg_time_eager.to_numpy()
                                
                                # Find the element-wise minimum
                                accl_avg_time = np.minimum(array_rndzvs, array_eager)
                                
                                # Print the element-wise minimum values
                                print(accl_avg_time)
                                print(union_indices)

                                array_eager_value = np.array([])
                                array_eager_idx = np.array([])
                                for idx, val in enumerate(accl_avg_time):
                                    if array_eager[idx] < array_rndzvs[idx]:
                                        array_eager_value = np.append(array_eager_value, [array_eager[idx]]) 
                                        array_eager_idx = np.append(array_eager_idx, [union_indices[idx]]) 
                                print(array_eager_value)
                                print(array_eager_idx)

                                plt.plot(union_indices, accl_avg_time,
                                            label=f'cclo-rndzvs-{stack}-{host}',
                                            linestyle=line_styles[host_idx % len(line_styles)],
                                            marker=markers[0 % len(markers)],
                                            linewidth=3, markersize=8, markeredgewidth=2)
                                
                                if array_eager_value.size != 0:
                                    plt.plot(array_eager_idx, array_eager_value,
                                            label=f'cclo-eager-{stack}-{host}',
                                            linestyle=line_styles[host_idx % len(line_styles)],
                                            marker=markers[1 % len(markers)],
                                            linewidth=3, markersize=8, markeredgewidth=2)
                            else:
                                if accl_avg_time_rndzvs is None and accl_avg_time_eager is not None:
                                    plt.plot(accl_avg_time_eager.index, accl_avg_time_eager,
                                            label=f'cclo-{stack}-{host}',
                                            linestyle=line_styles[host_idx % len(line_styles)],
                                            marker=markers[1 % len(markers)],
                                            linewidth=3, markersize=8, markeredgewidth=2)
                                if accl_avg_time_eager is None and accl_avg_time_rndzvs is not None:
                                    plt.plot(accl_avg_time_rndzvs.index, accl_avg_time_rndzvs,
                                            label=f'cclo-{stack}-{host}',
                                            linestyle=line_styles[host_idx % len(line_styles)],
                                            marker=markers[0 % len(markers)],
                                            linewidth=3, markersize=8, markeredgewidth=2)

                        # Find the min/max value among all ACCL lines for this host
                        max_accl_x = max(accl_max_x_list)
                        min_accl_x = min(accl_min_x_list)

                        # plot MPI RDMA lines with host data
                        if host == "host":
                            mpi_filter_df = mpi_filter_df[mpi_filter_df['size'] <= max_accl_x]
                            mpi_avg_time = mpi_filter_df["host_to_host"].reset_index(drop=True)
                            mpi_size = mpi_filter_df['size'].reset_index(drop=True) 
                            plt.plot(mpi_size, mpi_avg_time,
                                    label=f'mpi-rdma-host',
                                    linestyle='--',
                                    marker='^',
                                    color='red',
                                    linewidth=3, markersize=8, markeredgewidth=2)
                        
                        # # plot MPI RDMA lines with device data
                        if host == "device":
                            mpi_filter_df = mpi_filter_df[mpi_filter_df['size'] <= max_accl_x]
                            hd_filter_df = hd_filter_df[hd_filter_df['size'] <= max_accl_x]
                            mpi_avg_time = (mpi_filter_df["host_to_host"] + hd_filter_df['fpga_to_host'] + hd_filter_df['host_to_fpga']).reset_index(drop=True)
                            mpi_size = hd_filter_df['size'].reset_index(drop=True)
                            plt.plot(mpi_size, mpi_avg_time,
                                    label=f'mpi-rdma-device',
                                    linestyle='-',
                                    marker='^',
                                    color='red',
                                    linewidth=3, markersize=8, markeredgewidth=2)
        
                        # Process title/legend/fontsize
                        plt.xlabel('Size[B]',fontsize=18)
                        plt.ylabel('Latency[us]',fontsize=18)
                        plt.xscale('log', base=2)  # Set x-axis to log2 scale
                        plt.yscale('log', base=2)  # Set y-axis to log2 scale
                        # plt.title(f'Latency vs Size for {collective} ({nodes} nodes, {host} data)',fontsize=18)
                        plt.legend(fontsize=18)
                        plt.xticks(rotation=0, fontsize=18)
                        plt.yticks(fontsize=18)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'{collective}_rank_{nodes}_{host}_execution_time_merge.png'))
                        plt.close()


def generate_time_size_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, node_value):
    collective_values = accl_dataframes['collective'].unique()
    
    for collective in collective_values:
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)]
        mpi_df = mpi_dataframes.loc[(mpi_dataframes.index.get_level_values("collective") == collective)]
        hd_df = host_device_dataframes.loc[(host_device_dataframes.index.get_level_values("collective") == collective)]
        
        # plot time-size for each collective and each node
        if collective != 'sendrecv':
            number_of_nodes_values = accl_df['number_of_nodes'].unique()
            for nodes in number_of_nodes_values:
                if nodes == node_value:
                    accl_filter_df = accl_df[accl_df['number_of_nodes'] == nodes]
                    mpi_filter_df = mpi_df.loc[(mpi_df.index.get_level_values("number_of_nodes") == nodes)]
                    hd_filter_df = hd_df.loc[(hd_df.index.get_level_values("number_of_nodes") == nodes)]
                    # print(accl_filter_df)
                    # print(mpi_filter_df)  

                    # Create an empty list to store the min/max values for each line
                    accl_max_x_list = []
                    accl_min_x_list = []
                    
                    plt.figure()

                    # plot ACCL line, pick the lowest value from the eager and the rndzvs
                    for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')): 
                        for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
                            accl_avg_time_rndzvs = None
                            accl_avg_time_eager = None
                            for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                                if protoc == 'rndzvs':  
                                    accl_avg_time_rndzvs = protoc_group.groupby('size')['execution_time'].mean()
                                    accl_max_x = accl_avg_time_rndzvs.index.max()
                                    accl_min_x = accl_avg_time_rndzvs.index.min()
                                if protoc == 'eager':
                                    accl_avg_time_eager = protoc_group.groupby('size')['execution_time'].mean()
                                    accl_max_x = accl_avg_time_eager.index.max()
                                    accl_min_x = accl_avg_time_eager.index.min()
                                
                                # Store the maximum value for this ACCL line
                                accl_max_x_list.append(accl_max_x)
                                accl_min_x_list.append(accl_min_x)
                            
                            print(accl_avg_time_rndzvs)
                            print(accl_avg_time_eager)
                            if accl_avg_time_rndzvs is not None and accl_avg_time_eager is not None:
                                # Find the union of the indices and fill missing values with a large number
                                union_indices = accl_avg_time_rndzvs.index.union(accl_avg_time_eager.index)
                                accl_avg_time_rndzvs = accl_avg_time_rndzvs.reindex(union_indices, fill_value=np.inf)
                                accl_avg_time_eager = accl_avg_time_eager.reindex(union_indices, fill_value=np.inf)
                                
                                # Convert the series to numpy arrays
                                array_rndzvs = accl_avg_time_rndzvs.to_numpy()
                                array_eager = accl_avg_time_eager.to_numpy()
                                
                                # Find the element-wise minimum
                                accl_avg_time = np.minimum(array_rndzvs, array_eager)
                                
                                # Print the element-wise minimum values
                                print(accl_avg_time)
                                print(union_indices)

                                plt.plot(union_indices, accl_avg_time,
                                        label=f'cclo-{stack}-{host}',
                                        linestyle=line_styles[host_idx % len(line_styles)],
                                        marker=markers[protoc_idx % len(markers)],
                                        linewidth=3, markersize=8, markeredgewidth=2)
                            else:
                                if accl_avg_time_rndzvs is None and accl_avg_time_eager is not None:
                                    plt.plot(accl_avg_time_eager.index, accl_avg_time_eager,
                                            label=f'cclo-{stack}-{host}',
                                            linestyle=line_styles[host_idx % len(line_styles)],
                                            marker=markers[protoc_idx % len(markers)],
                                            linewidth=3, markersize=8, markeredgewidth=2)
                                if accl_avg_time_eager is None and accl_avg_time_rndzvs is not None:
                                    plt.plot(accl_avg_time_rndzvs.index, accl_avg_time_rndzvs,
                                            label=f'cclo-{stack}-{host}',
                                            linestyle=line_styles[host_idx % len(line_styles)],
                                            marker=markers[protoc_idx % len(markers)],
                                            linewidth=3, markersize=8, markeredgewidth=2)

                        # Find the min/max value among all ACCL lines for this host
                        max_accl_x = max(accl_max_x_list)
                        min_accl_x = min(accl_min_x_list)

                    # plot MPI RDMA lines with host data
                    mpi_filter_df = mpi_filter_df[mpi_filter_df['size'] <= max_accl_x]
                    mpi_avg_time = mpi_filter_df["host_to_host"].reset_index(drop=True)
                    mpi_size = mpi_filter_df['size'].reset_index(drop=True) 
                    plt.plot(mpi_size, mpi_avg_time,
                            label=f'mpi-rdma-host',
                            linestyle='--',
                            marker='^',
                            color='red',
                            linewidth=3, markersize=8, markeredgewidth=2)
                        
                    # plot MPI RDMA lines with device data
                    mpi_filter_df = mpi_filter_df[mpi_filter_df['size'] <= max_accl_x]
                    hd_filter_df = hd_filter_df[hd_filter_df['size'] <= max_accl_x]
                    mpi_avg_time = (mpi_filter_df["host_to_host"] + hd_filter_df['fpga_to_host'] + hd_filter_df['host_to_fpga']).reset_index(drop=True)
                    mpi_size = hd_filter_df['size'].reset_index(drop=True)
                    plt.plot(mpi_size, mpi_avg_time,
                            label=f'mpi-rdma-device',
                            linestyle='-',
                            marker='^',
                            color='red',
                            linewidth=3, markersize=8, markeredgewidth=2)
    
                    # Process title/legend/fontsize
                    plt.xlabel('Size[B]',fontsize=18)
                    plt.ylabel('Latency[us]',fontsize=18)
                    plt.xscale('log', base=2)  # Set x-axis to log2 scale
                    plt.yscale('log', base=2)  # Set y-axis to log2 scale
                    # plt.title(f'Latency vs Size for {collective} ({nodes} nodes, {host} data)',fontsize=18)
                    plt.legend(fontsize=18)
                    plt.xticks(rotation=0, fontsize=18)
                    plt.yticks(fontsize=18)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{collective}_rank_{nodes}_execution_time.png'))
                    plt.close()

def generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, size_kb):
    collective_values = accl_dataframes['collective'].unique()

    for collective in collective_values:
        if collective != 'sendrecv':
            accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)]
            mpi_df = mpi_dataframes.loc[(mpi_dataframes.index.get_level_values("collective") == collective)]
            hd_df = host_device_dataframes.loc[(host_device_dataframes.index.get_level_values("collective") == collective)]
            # Filter data for size
            accl_filter_df = accl_df[accl_df['size'] == size_kb * 1024]
            # print(accl_filter_df)
            mpi_filter_df = mpi_df[mpi_df['size'] == size_kb * 1024]
            # print(mpi_filter_df)
            hd_filter_df = hd_df[hd_df['size'] == size_kb * 1024]

            # plot separate data for host and device
            for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')):    

                # Create an empty list to store the min/max values for each line
                accl_max_x_list = []
                accl_min_x_list = []
                
                plt.figure()
                # plot ACCL line
                for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
                    for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                        accl_avg_time = protoc_group.groupby('number_of_nodes')['execution_time'].mean()
                        accl_std = protoc_group.groupby('number_of_nodes')['execution_time'].std()
                        print(accl_std)
                        accl_max_x = accl_avg_time.index.max()
                        accl_min_x = accl_avg_time.index.min()
                        # print(accl_avg_time)
                        plt.errorbar(accl_avg_time.index, accl_avg_time, yerr=accl_std,
                                label=f'cclo-{protoc}-{stack}-{host}',
                                linestyle=line_styles[host_idx % len(line_styles)],
                                marker=markers[protoc_idx % len(markers)],
                                linewidth=3, markersize=8, markeredgewidth=2)

                        # Store the maximum value for this ACCL line
                        accl_max_x_list.append(accl_max_x)
                        accl_min_x_list.append(accl_min_x)
                
                # Find the min/max value among all ACCL lines for this host
                max_accl_x = max(accl_max_x_list)
                min_accl_x = min(accl_min_x_list)
                        
                # plot MPI RDMA lines with host data
                if host == "host":
                    mpi_filter_df = mpi_filter_df.loc[(mpi_filter_df.index.get_level_values("number_of_nodes") <= max_accl_x)]
                    mpi_filter_df = mpi_filter_df.loc[(mpi_filter_df.index.get_level_values("number_of_nodes") >= min_accl_x)]
                    mpi_avg_time = mpi_filter_df["host_to_host"].reset_index(drop=True)
                    mpi_std_time = mpi_filter_df["host_to_host_std"].reset_index(drop=True)
                    number_of_nodes_values = mpi_filter_df.reset_index()['number_of_nodes']
                    print(mpi_avg_time)
                    print(mpi_std_time)
                    print(number_of_nodes_values)
                    plt.errorbar(number_of_nodes_values, mpi_avg_time, yerr=mpi_std_time,
                            label=f'mpi-rdma-host',
                            linestyle='--',
                            marker='^',
                            color='red',
                            linewidth=3, markersize=8, markeredgewidth=2)
                
                # # plot MPI RDMA lines with device data
                if host == "device":
                    mpi_filter_df = mpi_filter_df.loc[(mpi_filter_df.index.get_level_values("number_of_nodes") <= max_accl_x)]
                    mpi_filter_df = mpi_filter_df.loc[(mpi_filter_df.index.get_level_values("number_of_nodes") >= min_accl_x)]
                    hd_filter_df = hd_filter_df.loc[(hd_filter_df.index.get_level_values("number_of_nodes") <= max_accl_x)]
                    hd_filter_df = hd_filter_df.loc[(hd_filter_df.index.get_level_values("number_of_nodes") >= min_accl_x)]
                    mpi_avg_time = (mpi_filter_df["host_to_host"] + hd_filter_df['fpga_to_host'] + hd_filter_df['host_to_fpga']).reset_index(drop=True)
                    mpi_std_time = mpi_filter_df["host_to_host_std"].reset_index(drop=True)
                    number_of_nodes_values = mpi_filter_df.reset_index()['number_of_nodes']
                    print(mpi_avg_time)
                    print(mpi_std_time)
                    print(number_of_nodes_values)
                    plt.errorbar(number_of_nodes_values, mpi_avg_time, yerr=mpi_std_time,
                            label=f'mpi-rdma-device',
                            linestyle='-',
                            marker='^',
                            color='red',
                            linewidth=3, markersize=8, markeredgewidth=2)

                # Set labels, title, and legend
                plt.xlabel('Number of Nodes', fontsize=18)
                plt.ylabel('Latency [us]', fontsize=18)
                # plt.title(f'Time vs. Nodes for {collective} (Size: {size_kb}KB, Data: {host})', fontsize=18)
                plt.legend(fontsize=18, loc="upper left")
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.grid(True)

                # Save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{collective}_size_{size_kb}KB_{host}_execution_time.png'))
                plt.close()
            
if __name__ == "__main__":
    accl_log_dir = "../accl_results/results_reduce"  # Update this to the directory containing your log files
    output_dir = "../plots/"
    host_device_log_dir = "./host_device" 
    mpi_log_dir = "./results_eth_rdma" # point to mpi new results 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    accl_dataframes = read_accl_log_files(accl_log_dir)
    host_device_dataframes = read_host_device_log_files(host_device_log_dir)
    mpi_dataframes = read_mpi_rdma_log_files(mpi_log_dir)
    # print(mpi_dataframes)
    print(host_device_dataframes)
    # print(accl_dataframes)
    generate_throughput_plots(accl_dataframes, output_dir)
    generate_time_size_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 8)
    generate_time_size_plots_merge_eager_rndzvs(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 8)
    generate_sep_time_size_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 8)
    generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 4)
    generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 64)
    generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 128)
