import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from os.path import exists, join
from numba import jit


def main():
    num_procs = 12
    pool = Pool(num_procs, maxtasksperchild=1)
    members = np.arange(2, 14)
    lme_month_dir = "/glade/p/cesm0005/CESM-CAM5-LME/atm/proc/tseries/monthly/"
    lme_daily_dir = "/glade/p/cesm0005/CESM-CAM5-LME/atm/proc/tseries/daily/"
    out_path = "/glade/scratch/dgagne/ci_hackathon_2017/"
    output_variable = "PRECT"
    input_variables = ["PSL", "TROP_P", "TROP_T", "TS", "PS", "PRECT", "Q", "T", "U", "V", "Z3"]
    pres_vars = ["Q", "T", "U", "V", "Z3"]
    pres_levels = np.array([500, 850])
    input_month = 12
    precip_months = np.array([12, 1, 2])
    site_lon = -121.485556
    site_lat = 39.538889
    for member in members:
        for input_variable in input_variables:
            pool.apply_async(process_lme_member, (member, input_variable, input_month, lme_month_dir, out_path))
        pool.apply_async(calc_seasonal_precip, (site_lon, site_lat, member, output_variable,
                                                precip_months, lme_daily_dir, out_path))
    pool.close()
    pool.join()
    pool_2 = Pool(num_procs, maxtasksperchild=1)
    for member in members:
        pool_2.apply_async(precipitable_water, (member, input_month, out_path, out_path))
        for pres_var in pres_vars:
            pool_2.apply_async(interp_pressure, (pres_var, pres_levels,  member, input_month, out_path, out_path,))
    pool_2.close()
    pool_2.join()
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
        postind_ds = xr.open_dataset(postind_run_file)
        all_dates = pd.Series(xr.concat([preind_ds["date"], postind_ds["date"]], dim="time").astype("U8"))
        all_times = xr.concat([preind_ds["time"], postind_ds["time"]])
        all_years = all_dates.str[0:-4].astype(int)
        all_months = all_dates.str[-4:-2].astype(int)
        valid_months = np.where((all_months == input_month) & (all_years < 2005))
        all_data = xr.concat([preind_ds[input_variable], postind_ds[input_variable]], dim="time")
        month_data = all_data[valid_months]
        lat = preind_ds["lat"]
        lon = preind_ds["lon"] 
        out_filename = join(out_path,
                            "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                             input_variable,
                                                                                                             input_month))
        data_dict = {input_variable: month_data,
                                         "date": all_dates[valid_months],
                                         "year": all_years[valid_months]}
        for var in preind_ds.variables.keys():
            if var not in [input_variable, "date", "time", "datesec"]:
                data_dict[var] = preind_ds[var]
        out_data = xr.Dataset(data_vars=data_dict,
                              coords={"lat":lat, "lon": lon, "time": all_times[valid_months],
                                      "lev": preind_ds["lev"]})
        out_data.to_netcdf(out_filename, encoding={input_variable: {"dtype":"float32", 
                                                                    "zlib": True,
                                                                    "complevel": 2}})
        preind_ds.close()
        postind_ds.close()
    return


def calc_seasonal_precip(site_lon, site_lat, member, precip_var, precip_months, daily_path, out_path):
    preind_run_file = join(daily_path,
                           precip_var,
                           "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h1.{1:s}.08500101-18491231.nc".format(member,
                                                                                                      precip_var))
    postind_run_file = join(daily_path,
                            precip_var,
                            "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h1.{1:s}.18500101-20051231.nc".format(member,
                                                                                                      precip_var))
    if exists(preind_run_file) and exists(postind_run_file):
        preind_ds = xr.open_dataset(preind_run_file)
        postind_ds = xr.open_dataset(postind_run_file)
        all_dates = pd.Series(xr.concat([preind_ds["date"], postind_ds["date"]], dim="time").astype("U8"))
        # all_times = xr.concat([preind_ds["time"], postind_ds["time"]])
        all_years = all_dates.str[0:-4].astype(int)
        all_months = all_dates.str[-4:-2].astype(int)
        dec_years = np.copy(all_years[:])
        dec_years[all_months == 12] += 1
        valid_months = np.where(np.in1d(all_months, precip_months) & (dec_years > 850) & (dec_years < 2006))
        lat = preind_ds["lat"]
        lon = preind_ds["lon"]
        site_row = np.argmin(np.abs(lat - site_lat))
        site_col = np.argmin(np.abs(lon - site_lon))
        all_data = xr.concat([preind_ds[precip_var], postind_ds[precip_var]], dim="time")
        site_data = all_data[precip_var][valid_months, site_row, site_col]
        site_data_mm_day = site_data * 86400 * 1000
        daily_precip = pd.DataFrame({"year": dec_years[valid_months],
                                     "month": all_months[valid_months],
                                     precip_var: site_data_mm_day})
        seasonal_precip = daily_precip.groupby("year")[precip_var].sum()
        out_file = join(out_path, "djf_precip_lme_cam_{0:03d}_{1:s}_0851-2005.csv".format(member, precip_var))
        seasonal_precip.to_csv(out_file, index_label="Year")
    return


@jit()
def precipitable_water(member, input_month, data_path, out_path, pres_var="PS", q_var="Q"):
    pres_file = join(data_path,
                     "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                      pres_var,
                                                                                                      input_month))
    q_file = join(data_path,
                  "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                   q_var,
                                                                                                   input_month))
    if exists(pres_file) and exists(q_file):
        pres_ds = xr.open_dataset(pres_file)
        q_ds = xr.open_dataset(q_file)
        pw_field = np.zeros(pres_ds.shape, dtype=np.float32)
        ai = q_ds["hyai"].values
        bi = q_ds["hybi"].values
        p0 = q_ds["P0"].values
        for t in range(q_ds[q_var].shape[0]):
            q_field = q_ds[q_var][t].values
            pres_field = pres_ds[pres_var][t].values
            for (i, j), pres in np.ndenumerate(pres_field):
                pres_i_levels = ai * p0 + bi * pres
                pres_diffs = pres_i_levels[1:] - pres_i_levels[:-1]
                pw_field[t, i, j] = np.sum(pres_diffs * q_field[:, i, j]) / 9.81 # / 1000 kg/m^3 * 1000 mm / m
        data_dict = {"TMQ": pw_field}
        for var in q_ds.variables.keys():
            if var not in [q_var]:
                data_dict[var] = q_ds[var]
        pw_ds = xr.Dataset(data_vars=data_dict, coords={"lat": q_ds["lat"],
                                                        "lon": q_ds["lon"],
                                                        "time": q_ds["time"],
                                                         "lev": q_ds["lev"]})
        out_filestr = "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc"
        out_filename = join(out_path,
                            out_filestr.format(member,
                                               "TMQ",
                                               input_month))
        pw_ds.to_netcdf(out_filename, encoding={"TMQ": {"dtype":"float32",
                                                        "zlib": True,
                                                        "complevel": 2}})
        pres_ds.close()
        q_ds.close()
    return


@jit()
def interp_pressure(interp_var, pressure_levels, member, input_month, data_path, out_path, pres_var="PS"):
    pres_file = join(data_path,
                     "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                      pres_var,
                                                                                                      input_month))
    i_file = join(data_path,
                "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                 interp_var,
                                                                                                 input_month))
    if exists(pres_file) and exists(i_file):
        pres_ds = xr.open_dataset(pres_file)
        i_ds = xr.open_dataset(i_file)
        pres_level_field = np.zeros([len(pressure_levels)] + list(pres_ds.shape), dtype=np.float32)
        am = i_ds["hyam"].values
        bm = i_ds["hybm"].values
        p0 = i_ds["P0"].values
        for t in range(i_ds[interp_var].shape[0]):
            i_field = i_ds[interp_var][t].values
            pres_field = pres_ds[pres_var][t].values
            for (i, j), pres in np.ndenumerate(pres_field):
                pres_m_levels = am * p0 + bm * pres
                pres_level_field[:, t, i, j] = np.interp(pres_m_levels, i_field[:, i, j], pressure_levels * 100)
        for p, pressure_level in enumerate(pressure_levels):
            out_var = interp_var + "_{0:d}".format(int(pressure_level))
            data_dict = {out_var: pres_level_field[p]}
            for var in i_ds.variables.keys():
                if var not in [interp_var]:
                    data_dict[var] = i_ds[var]
            out_ds = xr.Dataset(data_vars=data_dict, coords={"lat": i_ds["lat"],
                                                             "lon": i_ds["lon"],
                                                             "time": i_ds["time"]})
            out_filestr = "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc"
            out_filename = join(out_path,
                                out_filestr.format(member,
                                                   out_var,
                                                   input_month))
            out_ds.to_netcdf(out_filename, encoding={out_var: {"dtype": "float32",
                                                               "zlib": True,
                                                               "complevel": 2}})
        pres_ds.close()
        i_ds.close()
    return


if __name__ == "__main__":
    main()
