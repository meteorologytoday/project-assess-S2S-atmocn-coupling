import pandas as pd
import argparse
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--date-rng', type=str, nargs=2, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()


    beg_dt = pd.Timestamp(args.date_rng[0])
    end_dt = pd.Timestamp(args.date_rng[1])
    output_dir = Path(args.output_dir)

    beg_dt_str = beg_dt.strftime("%Y-%m-%d")
    end_dt_str = end_dt.strftime("%Y-%m-%d")

    date_to_category_file = output_dir / f"year_month_group_mapping_{beg_dt_str:s}_{end_dt_str:s}.csv"
    category_file = output_dir / f"year_month_group_category_{beg_dt_str:s}_{end_dt_str:s}.csv"


    
    
    potential_dts = pd.date_range(beg_dt, end_dt, freq="D", inclusive="both")

    dts = []
    dt_to_category = []
    categories = []

    for _dt in potential_dts:

        if _dt.month not in [12, 1, 2]:
            
            continue
        
        category = _dt.strftime("%Y-%m")
        
        
        dts.append(_dt)    
        dt_to_category.append(_dt.strftime("%Y-%m"))

        if category not in categories:
            categories.append(category)
    

    df_date_to_category = pd.DataFrame.from_dict(
        dict(
            date = dts,
            category = dt_to_category,
        )
    )

    df_category = pd.DataFrame.from_dict(dict(
        category = categories,
    ))

    print("Output file: ", date_to_category_file)
    df_date_to_category.to_csv(date_to_category_file)
    print("Output file: ", category_file)
    df_category.to_csv(category_file)



