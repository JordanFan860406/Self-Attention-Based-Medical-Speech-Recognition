import os
# folders = ["E:\\user\\Desktop\\EE303\\論文研究\\2021討論資料\\Syllable\\sChiMeS-14-id-divie\\wav\\test\\"]
# file_dict = {}

# for folder in folders:
#     subfolders = os.listdir(folder)
#     for subfolder in subfolders:
#         if "S" in subfolder:
#             path = folder + subfolder
#             wavs = os.listdir(path)
#             index = 1
#             for wav in wavs:
#                 wav = wav.split(".")[0]
#                 newFileName = "BAC009" + subfolder + "w" + "0"*(4 - len(str(index))) + str(index)
#                 file_dict[newFileName] = wav
#                 index += 1
# # print(len(file_dict))
# with open("file_name_change.txt", "w", encoding="utf-8") as W:
#     for key in file_dict:
#         W.write(key + "\t" + file_dict[key] + "\n")
with open("E:\\user\\Desktop\\EE303\\論文研究\\2021討論資料\\code\\psChiMeS_5_wave_key_conformer.txt", "r", encoding="utf-8") as R:
    lines = R.readlines()

with open("E:\\user\\Desktop\\EE303\\論文研究\\2021討論資料\\code\\result\\psChiMeS_5_wave_key_conformer.txt", "w", encoding="utf-8") as W:
    for line in lines:
        line = line.split()
        file = line[-1].split("-")[1][:-1]
        result = "".join(line[:-1])
        print(file_dict[file], result)
        W.write(file_dict[file] + "\t" + result + "\n")