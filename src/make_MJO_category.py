import pandas as pd

if __name__ == "__main__":


    mjo_datafile = "data/omi.era5.1x.webpage.4023.txt.csv"
    mjo_date_to_category_file = "gendata/MJO_date_to_category.csv"
    mjo_category_file = "gendata/MJO_category.csv"

    print("Reading file: ", mjo_datafile)
    df_mjo_data = pd.read_csv(mjo_datafile)
    print(df_mjo_data)
        
    category_to_phase = { i : ("P%d" % i) for i in range(1, 9)}

    df_mjo_data["category"] = df_mjo_data.apply(lambda row: category_to_phase[int(row['phase'])] if row['magnitude'] >= 1 else "NoMJO", axis=1)


    mjo_date = pd.to_datetime(df_mjo_data['date'])


    for selected_months in [
        [10, 11, 12, 1, 2],
    ]:

        df_mjo_data = df_mjo_data[
            mjo_date.dt.month.isin(selected_months)
            & (mjo_date.dt.year >= 1998)
            & (mjo_date.dt.year <= 2017)
        ]

        df_mjo_date_to_category_file = df_mjo_data

        print("Output file: ", mjo_date_to_category_file)
        df_mjo_date_to_category_file.to_csv(mjo_date_to_category_file)


        df_MJO_category = pd.DataFrame.from_dict(dict(
            category = ["NoMJO", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
        ))

        print("Output file: ", mjo_category_file)
        df_MJO_category.to_csv(mjo_category_file)



