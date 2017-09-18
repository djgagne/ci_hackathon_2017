import numpy as np
import pandas as pd
import xarray as xr
from os.path import exists, join


def main():
    train_members = [2, 4, 6, 8]
    public_test_members = [3, 5, 7, 9]
    private_test_members = [10, 11, 12 , 13]
    small_train_members = [3, 5]
    small_test_members = [2, 4]
    small_variables = ["TS", "PSL", "TMQ"]
    variables = ["TS", "PSL", "TMQ", "U_500", "V_500", "Z3_500"]
    #variables = ["TS"]
    data_path = "/d1/dgagne/ci_hackathon_2017/"
    out_path = "/Users/dgagne/ramp-data/california_rainfall/data/"
    small_out_path = "/Users/dgagne/ramp-kits/california_rainfall_test/data/"
    generate_precip_labels(train_members, "train", data_path, out_path)
    generate_precip_labels(public_test_members, "public_test", data_path, out_path)
    generate_precip_labels(private_test_members, "test", data_path, out_path)
    generate_precip_labels(small_train_members, "train", data_path, small_out_path)
    generate_precip_labels(small_test_members, "test", data_path, small_out_path)
    #merge_ensemble_members(small_train_members, small_variables, "train", data_path, small_out_path, out_dtype="int8")
    #merge_ensemble_members(small_test_members, small_variables, "test", data_path, small_out_path, out_dtype="int8")
    #merge_ensemble_members(train_members, variables, "train", data_path, out_path)
    #merge_ensemble_members(public_test_members, variables, "public_test", data_path, out_path)
    merge_ensemble_members(private_test_members, variables, "test", data_path, out_path)
    return


def merge_ensemble_members(members, variables, collection, data_path, out_path, out_dtype="int16"):
    lme_file_template = "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.085012-200412.nc"
    nc_encoding = {}
    var_attrs = {"PSL": {"long_name": "Sea level pressure", "units": "Pa"},
                 "TS": {"long_name": "Surface temperature (radiative)", "units": "K"},
                 "TMQ": {"long_name": "Precipitable Water", "units": "mm"},
                 "U_500": {"long_name": "West-east wind component at 500 mb", "units": "m s-1"},
                 "V_500": {"long_name": "South-north wind component at 500 mb", "units": "m s-1"},
                 "Z3_500": {"long_name": "Geopotential height (above sea level)", "units": "m"},
    }
    for variable in variables:
        print(variable)
        var_coll = []
        nc_encoding[variable] = {"dtype": out_dtype, "_FillValue": -32767, "zlib": True, "complevel": 3}
        for member in members:
            print(member)
            nc_filename = join(data_path, lme_file_template.format(member, variable))
            print(nc_filename)
            if not exists(nc_filename):
                raise IOError
            var_coll.append(xr.open_dataset(nc_filename, decode_times=False))
            t_vals = pd.Series(var_coll[-1]["time"].values)
            print(t_vals[t_vals.duplicated()])
        merged_data = xr.concat(var_coll, dim="ens", data_vars="different")
        merged_data[variable].attrs = {**merged_data[variable].attrs, **var_attrs[variable]}
        merged_data.attrs["history"] = ""
        merged_data.attrs["case"] = ""
        print(merged_data[variable].attrs)
        scale_factor, add_offset = compute_scale_and_offset(merged_data[variable].min().values,
							    merged_data[variable].max().values,
                                                            int(out_dtype[3:]))
        print(scale_factor, add_offset)
        nc_encoding[variable]["scale_factor"] = scale_factor
        nc_encoding[variable]["add_offset"] = add_offset
        merged_data.to_netcdf(join(out_path, collection + "_{0}.nc".format(variable)), engine="netcdf4", encoding={variable:nc_encoding[variable]})
    
    return


def generate_precip_labels(members, collection, data_path, out_path):
    precip_file_template = "DJF_precip_lme_cam_{0:03d}_PRECT_0851-2005.csv"
    precip_data = []
    for e, ens in enumerate(members):
        precip_data.append(pd.read_csv(join(data_path, precip_file_template.format(ens)), 
                                       names=["Year", "Precip_{0:02d}".format(e)], index_col="Year"))
    all_precip_data = pd.concat(precip_data, axis=1)
    print("90th percentile", np.percentile(all_precip_data.values, 90))
    all_precip_avg = pd.DataFrame(np.where(all_precip_data.values >= 750, 1, 0), 
                                     index=all_precip_data.index, columns=all_precip_data.columns)
    all_precip_data.to_csv(join(out_path, collection + "_precip.csv"), index_label="Year")
    all_precip_avg.to_csv(join(out_path, collection + "_precip_90.csv"), index_label="Year")
    return

def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)

if __name__ == "__main__":
    main()
