print("Loading packages...")
import xarray as xr
import numpy as np
import argparse
import tool_fig_config
print("Done.")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input-file', type=str, help='Input directory.', required=True)
parser.add_argument('--output', type=str, help='Output directory.', default="")
parser.add_argument('--plot-lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
parser.add_argument('--plot-lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])
parser.add_argument('--no-display', action="store_true")
args = parser.parse_args()

print(args)

ds = xr.open_dataset(args.input_file)


regions = [
    "NW-PAC", "NE-PAC", "N-ATL", "NT-ATL", #"T-PAC", "T-ATL",
    "NDT-PAC", "NST-PAC", "NT-IND",
]

hatch_styles = [
    ".", "/", "-", "*", "|", "x", "o", "\\",
]

region_label_coords = {
    "N-PAC" :  (190.0, 45.0),
    "NW-PAC" : (155.0, 45.0),
    "NE-PAC" : (210.0, 45.0),
    "N-ATL" : (330.0, 45.0),
    "T-PAC" : (190.0, 15.0),
    "T-ATL" : (320.0, 15.0),
    "DT-PAC" :  (190.0, 0.0),
    "NST-PAC" :  (190.0, (15+30)/2),
    "T-IND" :  (75.0, 15.0),
    "NT-ATL" :   (320.0, 7.5),
    "NT-IND" :   (75.0,  7.5),
    "NDT-PAC" :  (190,   7.5),
}


print("Loading matplotlib...")

import matplotlib as mplt
if args.no_display:
    mplt.use("Agg")
else:
    mplt.use("TkAgg")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

print("done")

from scipy.stats import ttest_ind_from_stats


cent_lon = 210.0

plot_lon_l = args.plot_lon_rng[0]
plot_lon_r = args.plot_lon_rng[1]
plot_lat_b = args.plot_lat_rng[0]
plot_lat_t = args.plot_lat_rng[1]

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

ncol = 1
nrow = 1

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = 10,
    h = 5,
    wspace = 1.0,
    hspace = 0.5,
    w_left = 1.0,
    w_right = 2.2,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = ncol,
    nrow = nrow,
)


fig, ax = plt.subplots(
    nrow, ncol,
    figsize=figsize,
    subplot_kw=dict(projection=proj, aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

_ax = ax[0, 0]

coords = ds.coords

map_colors = [(0,0,0,0), (0,0,0,0.2), "g"]
cmap = matplotlib.colors.ListedColormap(map_colors)
bounds = np.arange(len(map_colors)+1)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)


for i in range(len(regions)):
    
    print( "norm(%d) = %f" % (i, norm(i),) )

for i, region in enumerate(regions):

    mask = ds["mask"].sel(region=region).to_numpy().astype(float)

    print(np.sum(mask))
    #mask[mask == 0] = np.nan   
    _mask = mask.copy()
    _mask[mask == 1] = 0.5
    _mask[mask == 0] = -0.5
 
    """
    mappable = _ax.contourf(
        coords["longitude"], coords["latitude"],
        _mask,
        levels=[0, 1, 2, 3],
        norm=norm,
        cmap=cmap, 
        transform=proj_norm,
    )
    """

    mappable = _ax.contourf(
        coords["longitude"], coords["latitude"],
        _mask,
        levels=[0, 1],
        colors="none",
        #norm=norm,
        #cmap=cmap,
        hatches = [hatch_styles[i],], 
        transform=proj_norm,
    )


    cs = _ax.contour(
        coords["longitude"], coords["latitude"],
        mask,
        levels=[0.5, 1.5],
        hatches=[".."],
        transform=proj_norm,
        colors="black",
    )

    if region in region_label_coords:
        region_label_coord = region_label_coords[region]
        _ax.text(region_label_coord[0], region_label_coord[1], region, transform=proj_norm, va="center", ha="center", backgroundcolor="white", size=15)
#plt.colorbar(mappable, ax=_ax)

    ## Plot the standard deviation
    #_contour = (data[0]["total_Estd"] + data[1]["total_Estd"]) / 2 / plot_info["factor"]
    #cs = _ax.contour(coords["longitude"], coords["latitude"], _contour, levels=plot_info["contour_levels"], colors="k", linestyles='-',linewidths=1, transform=proj_norm, alpha=0.8, zorder=10)
    #_ax.clabel(cs, fmt="%.1f")



# Plot the hatch to denote significant data
#_dot = np.zeros_like(pval)
#_dot[:] = np.nan

#_significant_idx =  (pval < args.pval_threshold) 
#_dot[ _significant_idx                 ] = 0.75
#_dot[ np.logical_not(_significant_idx) ] = 0.25

#cs = _ax.contourf(coords["longitude"], coords["latitude"], _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, "..."], transform=proj_norm)

# Remove the contour lines for hatches 
#for _, collection in enumerate(cs.collections):
#    collection.set_edgecolor((.2, .2, .2))
#    collection.set_linewidth(0.)

for __ax in [_ax, ]: 

    gl = __ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_top   = False
    gl.ylabels_right = False

    #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
    #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}

    __ax.set_global()
    #__ax.gridlines()
    __ax.coastlines(color='gray')
    __ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)



#cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
#cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)

#unit_str = "" if plot_info["unit"] == "" else " [ %s ]" % (plot_info["unit"],)
#cb.ax.set_ylabel("%s\n%s" % (plot_info["label"], unit_str), size=25)


if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)

print("Finished.")

