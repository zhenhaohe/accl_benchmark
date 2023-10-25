import pandas as pd
import matplotlib.pyplot as plt
import os
from os import error, listdir, name, terminal_size
from os.path import isfile, join
from numpy import average, mean, std
import numpy as np

# Set the maximum number of rows to be displayed
pd.options.display.max_rows = 2000  # Change 100 to the desired number of rows you want to display

line_styles = ['-', '--', ':']  # Different line styles for host values
markers = ['o', 'D', 's']  # Different markers for protoc values
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

def plot_additive(title, x_datas, y_datas, y_series_labels, y_styles=None, logx=True, x_label='Size [B]', y_label='Latency [μs]', legend_loc=None):
    if not(y_styles):
        y_styles = [None for _ in range(len(y_series_labels))]

    fig, ax = plt.subplots(figsize=(9,5))
    series  = []
    x_data = x_datas[0]
    lines = ax.stackplot(x_data, y_datas, labels=y_series_labels)

    plt.grid(True)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label,  fontsize=20)
    # ax.axis(ymin=1)
    ax.set_xlim(xmin=min(x_data), xmax=max(x_data))
    #ax.set_title(title)

    if logx:
        ax.set_xscale('log', base=2)

    if legend_loc is None:
        ax.legend(loc="upper left", handlelength=4, handles=lines[1:] + [lines[0]])
    else:
        ax.legend(loc=legend_loc, fontsize=14, handlelength=4, handles=lines[1:] + [lines[0]])

    if x_label == "Message Size":
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: sizeof_fmt(y)))
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    ax.set_xlabel(x_label, fontsize=20)
    plt.savefig(f"{title}.png", format='png', bbox_inches='tight')

def generate_throughput_plots(accl_dataframes, output_dir):
    
    accl_df = accl_dataframes[(accl_dataframes['collective'] == 'sendrecv')]

    # plot the throughput of send/recv
    accl_filter_df = accl_df[accl_df['rank_id'] == 1]
    plt.figure(figsize=(9,5))
    i=0
    for host_idx, (host, host_group) in enumerate(accl_filter_df.groupby('host')):
        for stack_idx, (stack, stack_group) in enumerate(host_group.groupby('stack')):   
            for protoc_idx, (protoc, protoc_group) in enumerate(stack_group.groupby('protoc')):
                accl_avg_tput = protoc_group.groupby('size')['throughput'].mean()
                # if host == 'host':
                plt.plot(accl_avg_tput.index, accl_avg_tput,
                        label=f'cclo-{protoc}-{stack}-{host}',
                        linestyle=line_styles[host_idx % len(line_styles)],
                        marker=markers[protoc_idx % len(markers)],
                        color=colors[i % len(colors)],
                        linewidth=3, markersize=8, markeredgewidth=2)
                i=i+1
    # OpenMPI + RDMA
    bufsize = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304])
    thr = np.array([0.7726489309353635, 1.2779287701841844, 2.7302214434400196, 4.587270185413581, 8.350978141370119, 15.087549850518753, 24.924359525455834, 37.81030581465406, 52.47362080324127, 66.69514730661872, 77.53762086733578, 84.44661209626463, 87.91596214714582, 90.26373933885196])
    plt.plot(bufsize, thr,
            label=f'mpi-rdma-host',
            linestyle='--',
            marker='^',
            color='red',
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
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)  & (accl_dataframes['size'] <= 262144)]
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
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)  & (accl_dataframes['size'] <= 262144)]
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
        accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)  & (accl_dataframes['size'] <= 262144)]
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
            accl_df = accl_dataframes[(accl_dataframes['collective'] == collective)  & (accl_dataframes['size'] <= 262144)]
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

def latency_breakdown(error=False):
    # broadcast
    series_label = []
    series_y     = []
    series_x     = []
    styles       = []
    stdevs       = []
    breakpoint = 11

    bufsize = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304])
    fh = [13.888764000000002, 11.059779999999998, 11.149212, 11.237948000000001, 11.490864000000002, 12.432864000000002, 13.788632, 16.456580000000002, 22.034295999999998, 32.552080000000004, 54.42410399999999, 107.308596, 196.2307, 368.71310000000005]
    fh_std = [5.512100217186186, 0.34413273543794126, 0.27489813214352704, 0.3652826430259177, 0.2500485023030532, 0.25619429639240604, 0.2586189950023006, 0.2624837434966211, 0.42741572313615245, 0.3893609656860839, 1.104373464541773, 1.4193851939427862, 5.377396685571933, 1.6945929570253706]
    mpi = [13.114632, 13.614564, 12.894744, 18.374896, 19.036768, 23.279864, 29.262196, 49.220988000000006, 88.575212, 166.908944, 325.63450800000004, 643.086088, 1277.28476, 2545.22112]
    mpi_std = [1.1602307290259122, 0.5146638571961314, 0.7359798628114765, 0.3887091524314806, 0.6079382025962835, 0.46755179125311885, 1.1156919850854894, 0.5207615921475007, 0.44911810368320715, 2.670622025832934, 0.6173208970511167, 0.9972786853512857, 1.3918805057906407, 0.5326585638098759]
    hf = [15.201824, 11.976196, 12.092144000000001, 12.191952, 12.501655999999999, 13.331800000000001, 17.021756, 17.614883999999996, 23.30192, 35.821708, 59.62975599999999, 124.534576, 211.782104, 408.62922399999997]
    hf_std = [6.651402709881879, 0.35229712116904954, 0.4064142065233448, 0.42474470178684987, 0.641864279785065, 0.5117330475941532, 5.375818835532313, 0.5155476219943216, 0.45789663637113576, 0.5841435223778485, 1.8558411528102294, 108.11323790276667, 3.1424371098216106, 3.99487642134572]
    nop = [12 for _ in bufsize]
    nop_std = [4.42 for _ in bufsize]
    streaming = [b / 32 * 0.0022 for b in bufsize]
    zero = [0 for _ in bufsize]

    # series_label.append(f"Kernel invocation")
    # series_y.append(np.array(nop))
    # series_x.append(bufsize)
    # stdevs.append(np.array(nop_std))
    # styles.append(f"C1+-")

    # series_label.append(f"Stream flushing")
    # series_y.append(np.array(streaming))
    # series_x.append(bufsize)
    # stdevs.append(np.array(zero))
    # styles.append(f"C2+-")

    series_label.append(f"FPGA HBM to host DDR")
    series_y.append(np.array(fh))
    series_x.append(bufsize)
    stdevs.append(np.array(fh_std))
    styles.append(f"C3+-")

    series_label.append(f"MPI using RoCE")
    series_y.append(np.array(mpi))
    series_x.append(bufsize)
    stdevs.append(np.array(mpi_std))
    styles.append(f"C4+-")

    series_label.append(f"Host DDR to FPGA HBM")
    series_y.append(np.array(hf))
    series_x.append(bufsize)
    stdevs.append(np.array(hf_std))
    styles.append(f"C5+-")

    plot_additive("latency_breakdown_bcast_stacked_nr_8", [s[:breakpoint] for s in series_x], [s[:breakpoint] for s in series_y], series_label, styles, y_label='Latency [μs]', logx=True, legend_loc="upper left")

if __name__ == "__main__":
    accl_log_dir = "../accl_results/results_send"  # Update this to the directory containing your log files
    output_dir = "../plots/"
    host_device_log_dir = "./host_device" 
    mpi_log_dir = "./results_eth_rdma" # point to mpi new results 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    accl_dataframes = read_accl_log_files(accl_log_dir)
    host_device_dataframes = read_host_device_log_files(host_device_log_dir)
    mpi_dataframes = read_mpi_rdma_log_files(mpi_log_dir)
    # print(mpi_dataframes)
    # print(host_device_dataframes)
    # print(accl_dataframes)
    generate_throughput_plots(accl_dataframes, output_dir)
    generate_time_size_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 8)
    generate_time_size_plots_merge_eager_rndzvs(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 8)
    generate_sep_time_size_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 8)
    generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 4)
    generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 64)
    generate_time_nodes_plots(accl_dataframes, mpi_dataframes, host_device_dataframes, output_dir, 128)
    latency_breakdown()
