import numpy as np
import re

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', type=str, help='Input directory.', default="raw_model_version_dates.txt")
    parser.add_argument('--output-file', type=str, help='Input directory.', default="model_version_dates.txt")
    args = parser.parse_args()
    print(args)
    
    raw_file = args.input_file
    output_file = args.output_file

    date_fmt = re.compile(r'^\s*(\d{4}-\d{2}-\d{2})\s*$')


    with open(output_file, "w") as out_f:
        with open(raw_file, "r") as in_f:
            for s in in_f.readlines():
                m = date_fmt.match(s)
                if m is not None:
                    print(m.group(1))
                    out_f.write(m.group(1))
                    out_f.write("\n")

