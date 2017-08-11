import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
from os.path import exists, join
from numba import jit
from os import mkdir
import traceback
from netCDF4 import num2date, date2num


def main():
    num_procs = 36
    pool = Pool(num_procs, maxtasksperchild=1)
    members = np.arange(2, 14)
    lme_month_dir = "/glade/p/cesm0005/CESM-CAM5-LME/atm/proc/tseries/monthly/"
    lme_daily_dir = "/glade/p/cesm0005/CESM-CAM5-LME/atm/proc/tseries/daily/"
    out_path = "/glade/scratch/dgagne/ci_hackathon_2017/"
    if not exists(out_path):
        mkdir(out_path)
    output_variable = "PRECT"
    input_variables = ["PSL", "TROP_P", "TROP_T", "TS", "PS", "Q", "T", "U", "V", "Z3"]
    pres_vars = ["Q", "T", "U", "V", "Z3"]
    pres_levels = np.array([500, 850])
    input_month = 12
    precip_months = np.array([12, 1, 2])
    site_lon = -121.485556 + 360
    site_lat = 39.538889
    #for member in members:
        #for input_variable in input_variables:
        #    pool.apply_async(process_lme_member, (member, input_variable, input_month, lme_month_dir, out_path))
        #pool.apply_async(calc_seasonal_precip, (site_lon, site_lat, member, output_variable,
        #                                        precip_months, lme_daily_dir, out_path))
    for member in members:
        pool.apply_async(precipitable_water, (member, input_month, out_path, out_path))
        for pres_var in pres_vars:
            pool.apply_async(interp_pressure, (pres_var, pres_levels,  member, input_month, out_path, out_path,))
    pool.close()
    pool.join()
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
    print("Starting {0:d} {1}".format(member, input_variable))
    try:
        if exists(preind_run_file) and exists(postind_run_file):
            preind_ds = xr.open_dataset(preind_run_file, decode_times=False)
            postind_ds = xr.open_dataset(postind_run_file, decode_times=False)
            all_dates = pd.Series(xr.concat([preind_ds["date"], postind_ds["date"]], dim="time").astype("U8"))
            preind_time = preind_ds["time"]
            postind_time = postind_ds["time"]
            preind_datetime = num2date(preind_time, preind_time.units, calendar=preind_time.calendar)
            postind_datetime = num2date(preind_time, postind_time.units, calendar=postind_time.calendar)
            all_datenums = date2num(np.concatenate([preind_datetime, postind_datetime]), preind_time.units, 
                                    calendar=preind_time.calendar)
            all_years = all_dates.str[0:-4].astype(int)
            all_months = all_dates.str[-4:-2].astype(int)
            valid_months = np.where((all_months == input_month) & (all_years < 2005))
            valid_times = xr.DataArray(all_datenums[valid_months], 
                                       coords={"time": all_datenums[valid_months]}, dims="time", attrs=preind_time.attrs)
            print(valid_times)
            all_data = xr.concat([preind_ds[input_variable], postind_ds[input_variable]], dim="time")
            month_data = all_data[valid_months].values
            lat = preind_ds["lat"]
            lon = preind_ds["lon"] 
            out_filename = join(out_path,
                                "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                                input_variable,
                                                                                                                input_month))
            data_dict = {input_variable: (preind_ds[input_variable].dims, month_data),
                         "date": (("time", ), all_dates.values[valid_months]),
                         "year": (("time", ), all_years.values[valid_months])}
            exclude_var_list = [input_variable, "date", "time", "datesec", "lat", "lon", "lev"]
            for var in preind_ds.variables.keys():
                if (var not in exclude_var_list) and ("time" not in preind_ds[var].dims):
                    data_dict[var] = preind_ds[var]
            print("Saving {0:d} {1}".format(member, input_variable))
            out_data = xr.Dataset(data_vars=data_dict,
                                  coords={"lat":lat, "lon": lon, "time": valid_times,
                                          "lev": preind_ds["lev"]}, attrs=preind_ds.attrs)
            out_data.to_netcdf(out_filename, encoding={input_variable: {"dtype":"float32", 
                                                                        "zlib": True,
                                                                        "complevel": 2}})
            preind_ds.close()
            postind_ds.close()
        else:
            print(str(member) + " " + input_variable + " does not exist")
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


def calc_seasonal_precip(site_lon, site_lat, member, precip_var, precip_months, daily_path, out_path):
    """
    Calculate the total precipitation over a set of months at the grid point nearest a specified site.

    Args:
        site_lon: longitude of the location where precipitation is being extracted
        site_lat: latitude of the location where precipitation is being extracted
        member: Number of the LME member
        precip_var: Precip variable being used. Recommend PRECT
        precip_months: Array of months being aggregated for seasonal precip
        daily_path: Path to daily LME files
        out_path: Path to where precip csv files are output.

    Returns:

    """
    try:
        preind_run_file = join(daily_path,
                            precip_var,
                            "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h1.{1:s}.08500101-18491231.nc".format(member,
                                                                                                        precip_var))
        postind_run_file = join(daily_path,
                                precip_var,
                                "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h1.{1:s}.18500101-20051231.nc".format(member,
                                                                                                        precip_var))
        print("Starting precip calc for {0:d}".format(member))
        if exists(preind_run_file) and exists(postind_run_file):
            preind_ds = xr.open_dataset(preind_run_file, decode_times=False)
            postind_ds = xr.open_dataset(postind_run_file, decode_times=False)
            all_dates = pd.Series(xr.concat([preind_ds["date"], postind_ds["date"]], dim="time").astype("U8"))
            #preind_time = preind_ds["time"]
            #postind_time = postind_ds["time"]
            #preind_datetime = num2date(preind_time, preind_time.units, calendar=preind_time.calendar)
            #postind_datetime = num2date(preind_time, postind_time.units, calendar=postind_time.calendar)
            #all_datenums = date2num(np.concatenate([preind_datetime, postind_datetime]), preind_time.units, 
            #                        calendar=preind_time.calendar)
            #all_times = xr.DataArray(all_datenums, coords={"time": all_datenums}, dims="time", attrs=preind_time.attrs)

            # all_times = xr.concat([preind_ds["time"], postind_ds["time"]])
            all_years = all_dates.str[0:-4].values.astype(int)
            all_months = all_dates.str[-4:-2].values.astype(int)
            dec_years = np.copy(all_years[:])
            dec_years[all_months == 12] += 1
            valid_months = np.where(np.in1d(all_months, precip_months) & (dec_years > 850) & (dec_years < 2006))[0]
            lat = preind_ds["lat"].values
            lon = preind_ds["lon"].values
            site_row = np.argmin(np.abs(lat - site_lat))
            site_col = np.argmin(np.abs(lon - site_lon))
            print(site_row, site_col)
            print(precip_var)
            all_data = np.concatenate([preind_ds[precip_var][:, site_row, site_col].values, 
                                       postind_ds[precip_var][:, site_row, site_col].values])
            site_data = all_data[valid_months]
            site_data_mm_day = site_data * 86400 * 1000
            daily_precip = pd.DataFrame({"year": dec_years[valid_months],
                                        "month": all_months[valid_months],
                                        precip_var: site_data_mm_day})
            seasonal_precip = daily_precip.groupby("year")[precip_var].sum()
            month_to_letter = {1: "J", 2: "F", 3: "M", 4: "A", 5: "M", 6: "J",
                            7: "J", 8: "A", 9: "S", 10: "O", 11: "N", 12: "D"}
            season_name = "".join([month_to_letter[x] for x in precip_months])
            out_file = join(out_path, "{0:s}_precip_lme_cam_{1:03d}_{2:s}_0851-2005.csv".format(season_name,
                                                                                                member, precip_var))
            
            print("Saving {0:d} precip".format(member))
            seasonal_precip.to_csv(out_file, index_label="Year")
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


def precipitable_water(member, input_month, data_path, out_path, pres_var="PS", q_var="Q"):
    try:
        pres_file = join(data_path,
                        "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                        pres_var,
                                                                                                        input_month))
        q_file = join(data_path,
                    "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                    q_var,
                                                                                                    input_month))
        print(pres_file, exists(pres_file))
        print(q_file, exists(q_file))
        if exists(pres_file) and exists(q_file):
            print("Starting precipitable water {0:d}".format(member))
            pres_ds = xr.open_dataset(pres_file, decode_times=False)
            q_ds = xr.open_dataset(q_file, decode_times=False)
            pw_field = np.zeros(pres_ds[pres_var].shape, dtype=np.float32)
            ai = q_ds["hyai"].values
            bi = q_ds["hybi"].values
            p0 = q_ds["P0"].values
            for t in range(q_ds[q_var].shape[0]):
                print("PW {0:02d} {1:03d}".format(member, t))
                q_field = q_ds[q_var][t].values
                pres_field = pres_ds[pres_var][t].values
                for (i, j), pres in np.ndenumerate(pres_field):
                    pres_i_levels = ai * p0 + bi * pres
                    pres_diffs = pres_i_levels[1:] - pres_i_levels[:-1]
                    pw_field[t, i, j] = np.sum(pres_diffs * q_field[:, i, j]) / 9.81 # / 1000 kg/m^3 * 1000 mm / m
            data_dict = {"TMQ": (("time", "lat", "lon"), pw_field)}
            for var in q_ds.variables.keys():
                if var not in [q_var, "lat", "lon", "time", "lev"]:
                    data_dict[var] = q_ds[var]
            print("Saving PW {0:02d}".format(member))
            pw_ds = xr.Dataset(data_vars=data_dict, coords={"lat": q_ds["lat"],
                                                            "lon": q_ds["lon"],
                                                            "time": q_ds["time"],
                                                            "lev": q_ds["lev"]}, attrs=q_ds.attrs)
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
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


def interp_pressure(interp_var, pressure_levels, member, input_month, data_path, out_path, pres_var="PS"):
    try:
        pres_file = join(data_path,
                        "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                        pres_var,
                                                                                                        input_month))
        i_file = join(data_path,
                    "b.e11.BLMTRC5CN.f19_g16.{0:03d}.cam.h0.{1:s}.0850{2:02d}-2004{2:02d}.nc".format(member,
                                                                                                    interp_var,
                                                                                                    input_month))
        if exists(pres_file) and exists(i_file):
            print("Starting pressure interp {0:02d} {1}".format(member, interp_var))
            pres_ds = xr.open_dataset(pres_file, decode_times=False)
            i_ds = xr.open_dataset(i_file, decode_times=False)
            pres_level_field = np.zeros([len(pressure_levels)] + list(pres_ds[pres_var].shape), dtype=np.float32)
            am = i_ds["hyam"].values
            bm = i_ds["hybm"].values
            p0 = i_ds["P0"].values
            for t in range(i_ds[interp_var].shape[0]):
                print("Pressure interp {0:02d} {1} t={2:03d}".format(member, interp_var, t))
                i_field = i_ds[interp_var][t].values
                pres_field = pres_ds[pres_var][t].values
                for (i, j), pres in np.ndenumerate(pres_field):
                    pres_m_levels = am * p0 + bm * pres
                    pres_level_field[:, t, i, j] = np.interp(pressure_levels * 100, pres_m_levels, i_field[:, i, j]) 
            print("Saving pressure interp {0:02d} {1}".format(member, interp_var))
            for p, pressure_level in enumerate(pressure_levels):
                out_var = interp_var + "_{0:d}".format(int(pressure_level))
                data_dict = {out_var: (("time", "lat", "lon"), pres_level_field[p])}
                for var in i_ds.variables.keys():
                    if var not in [interp_var, "lat", "lon", "time", "lev"]:
                        data_dict[var] = i_ds[var]
                out_ds = xr.Dataset(data_vars=data_dict, coords={"lat": i_ds["lat"],
                                                                 "lon": i_ds["lon"],
                                                                 "time": i_ds["time"],
                                                                 "lev": i_ds["lev"]}, attrs=i_ds.attrs)
                out_ds.attrs["pressure_level"] = str(pressure_level)
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
    except Exception as e:
        print(traceback.format_exc())
        raise e
    return


if __name__ == "__main__":
    main()
