import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import h5py

def convert_hdf5_to_parquet(h5_file, parquet_file, key_of_h5_file):
    with h5py.File(h5_file, "r") as f:
        ds_arr = f[key_of_h5_file][()]  # returns as a numpy array

    #1000 x 128
    print(f"Print some information of test dataset file shape: {ds_arr.shape}")
    print(ds_arr)
    ids = [i for i in range(len(ds_arr))]
    pa.array(ids)
    dataset = pa.array([ emb for emb in ds_arr ])
    cols = ['id', 'emb']
    table = pa.Table.from_arrays([ids, dataset], names=cols)

    pq.write_table(table, parquet_file)
    print(f"File is written here: {parquet_file}")

def read_file(file_to_read):
    df = pd.read_parquet(file_to_read)
    print(f"Reading the parquet file: {file_to_read}")
    print(df.head()) # prints the first few rows of the DataFrame

if __name__ == "__main__":
    h5_file = sys.argv[1]
    parquet_file = sys.argv[2]
    key_of_h5_file = sys.argv[3]
    convert_hdf5_to_parquet(h5_file, parquet_file, key_of_h5_file)
    read_file(parquet_file)