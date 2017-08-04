import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from multiprocessing import Pool
from os.path import exists, join

def main():
    num_procs = 12
    pool = Pool(num_procs)
    members = []
    lme_month_dir = "/glade/p/cesm0005/CESM-CAM5-LME/atm/proc/tseries/monthly/"
    lme_daily_dir = "/glade/p/cesm0005/CESM-CAM5-LME/atm/proc/tseries/daily/"
    output_variable = "PRECT"
    input_variables = ["PSL"]
    return


def process_lme_member(member, input_variable, input_month, monthly_path, out_path):
    """
    Load model output from single variable file of last millennium ensemble member, extract relevant months,
    and save to separate, smaller netCDF file.

    Args:
        member (int): numeric identity of member
        input_variable (str): name of the input variable being extracted
        input_month (int): month being extracted from monthly averaged data
        monthly_path (str): path to top level of monthly averaged LME files
        out_path (str): path to where subset files are saved.

    Returns:

    """
    preind_run_file = join(monthly_path,
                           input_variable,
                           "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.085001-184912.nc".format(member,
                                                                                                  input_variable))
    postind_run_file = join(monthly_path,
                           input_variable,
                           "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.185001-200512.nc".format(member,
                                                                                                  input_variable))
    if exists(preind_run_file) and exists(postind_run_file):
        preind_ds = xr.open_dataset(preind_run_file)
        preind_dates = pd.Series(preind_ds["date"][:].values.astype("U8"))
        preind_months = preind_dates.str[-4:-2].astype(int)
        preind_month_inds = np.where(preind_months == input_month)[0]
        preind_data = preind_ds[input_variable][preind_month_inds].values


    return

if __name__ == "__main__":
    main()
