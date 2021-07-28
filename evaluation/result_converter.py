import os
import argparse

parser = argparse.ArgumentParser(description="change testing result format")
parser.add_argument("--input_file_path", default="evaluation/psChiMeS_14_baseline_conformer.txt", type=str, help="input your original file path")
parser.add_argument("--output_file_path", default="evaluation/psChiMeS_14_baseline_conformers.txt", type=str, help="input your output file path")
args = parser.parse_args()

# load filename format
file_dict = {}
with open("manifest/file_name_change.txt") as R:
    lines = R.readlines()
for line in lines:
    line = line.split("\n")[0].split("\t")
    file_dict[line[0]] = line[1]
# print(format_dict)

# read orignial result file
with open(args.input_file_path, "r", encoding="utf-8") as R:
    lines = R.readlines()

# change espnet result format to original format
with open(args.output_file_path, "w", encoding="utf-8") as W:
    for line in lines:
        line = line.split()
        file = line[-1].split("-")[1][:-1]
        result = "".join(line[:-1])
        # print(file_dict[file], result)
        W.write(file_dict[file] + "\t" + result + "\n")