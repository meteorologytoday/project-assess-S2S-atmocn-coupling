import numpy as np

default_fmt = "%.1f"

def pretty_lat(lat, fmt=None):

    if fmt is None:
        fmt = default_fmt

    if np.abs(lat) > 90:
        raise Exception("Absolute value of latitude cannot be greater than 90. We have: ", str(lat))
    
    if lat == 0:
        lat_str = ( ("%sE" % fmt) % lat )
    
    elif lat > 0:
        lat_str = ( ("%sN" % fmt) % lat )

    else:
        lat_str = ( ("%sS" % fmt) % (- lat,) )


    return lat_str

def pretty_lon(lon, fmt=None):
 
    if fmt is None:
        fmt = default_fmt

   
    lon = lon % 360.0


    if lon < 180.0:
        lon_str = ( ("%sE" % fmt) % lon )

    else:
        lon_str = ( ("%sW" % fmt) % (360.0 - lon,) )

    return lon_str

def pretty_latlon(lat=None, lon=None, fmt=None):

    if fmt is None:
        fmt = default_fmt

    lat_str = None
    lon_str = None

    if lat is not None:
        lat_str = pretty_lat(lat, fmt=fmt)

    if lon is not None:
        lon_str = pretty_lon(lon, fmt=fmt)

    return dict(lat=lat_str, lon=lon_str)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
                        prog = 'make_ECCC_AR_objects.py',
                        description = 'Postprocess ECCO data (Mixed-Layer integrated).',
    )

    parser.add_argument('--func', type=str, required=True, choices=["box",])
    parser.add_argument('--fmt', type=str, default="default", choices=["int", "default"])
    parser.add_argument('--lat-rng', type=float, nargs=2, default=None)
    parser.add_argument('--lon-rng', type=float, nargs=2, default=None)
    args = parser.parse_args()
    #print(args)
    
    result = "UNKNOWN"
    
    if args.fmt == "int":
        default_fmt = "%d"

    if args.func == "box":
        
        result = "%s-%s_%s-%s" % (
            pretty_lat(args.lat_rng[0]), pretty_lat(args.lat_rng[1]),
            pretty_lon(args.lon_rng[0]), pretty_lon(args.lon_rng[1]),
        )

    print(result)
        
