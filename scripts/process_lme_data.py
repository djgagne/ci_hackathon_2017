import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from multiprocessing import Pool


def main():
    num_procs = 12
    pool = Pool(num_procs)
    members = []
    output_variable = "PRECT"
    input_variables = ["PSL"]
    return


def process_lme_member(member, input_variables, output_variable, input_month, output_months, monthly_path, daily_path, out_path):
    return

if __name__ == "__main__":
    main()
