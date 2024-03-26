#!/bin/bash



fig_dir=fig_ARoccur_stat

mkdir -p $fig_dir


box_params=(
    "REGION_06" 30 40 170 180

)
box_params=(
    "REGION_01" 30 40 120 130
    "REGION_02" 30 40 130 140
    "REGION_03" 30 40 140 150
    "REGION_04" 30 40 150 160
    "REGION_05" 30 40 160 170
    "REGION_06" 30 40 170 180
    "REGION_07" 30 40 180 190
    "REGION_08" 30 40 190 200
    "REGION_09" 30 40 200 210
    "REGION_10" 30 40 210 220
    "REGION_11" 30 40 220 230
    "REGION_12" 30 40 230 240
)


    #"REGION_12" 30 40 230 240

nparams=5
for (( i=0 ; i < $(( ${#box_params[@]} / $nparams )) ; i++ )); do

    box_name="${box_params[$(( i * $nparams + 0 ))]}"
    lat_s="${box_params[$(( i * $nparams + 1 ))]}"
    lat_n="${box_params[$(( i * $nparams + 2 ))]}"
    lon_w="${box_params[$(( i * $nparams + 3 ))]}"
    lon_e="${box_params[$(( i * $nparams + 4 ))]}"

    echo "# Doing Box name: $box_name"
    echo "## Lat : [${lat_s}, ${lat_n}]"
    echo "## Lon : [${lon_w}, ${lon_e}]"

    region=$( python3 pretty_latlon.py --func box --fmt int --lat-rng $lat_s $lat_n --lon-rng $lon_w $lon_e )
    input_dir="analysis_ARoccurence_stat/category/$region"

    output_dir=$fig_dir/$region
    mkdir -p $output_dir
    echo $region

    for months in "10" "11" "12" "01" "02" "03" "12-01-02" "10-11-12-01-02-03"; do

        input_file1="$input_dir/ECCC-S2S_GEPS5_ARoccur-stat-category_${months}.nc"
        input_file2="$input_dir/ECCC-S2S_GEPS6_ARoccur-stat-category_${months}.nc"

        python3 plot_BS.py \
            --input-files $input_file1 $input_file2 \
            --dataset-names GEPS5 GEPS6 \
            --output $output_dir/ARoccur-stat-category_${months}.png \
            --title "[$region] Month: $months" \
            --no-display

    done

done
 
