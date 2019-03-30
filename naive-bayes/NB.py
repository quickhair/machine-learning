# Minh Nguyen
# minh.nguyen@hunter.cuny.edu

import os
import time
from PreProcess import naiveB

start = time.time()

#labels = ["act", "com"]
#directory = "small-train/"
labels = ["pos", "neg"]
directory = "train/"

print ()
print ("Training...")
nb = naiveB()
for i in labels:
    nb.training(str(directory) + str(i), i)
print ("Done!")

#directory = "small-test/"
directory = "test/"

total = 0
pos_index = 0
neg_index = 0
correct_pred = 0

output_file = open("movie-review-BOW.NB", "w")
print ()
print ("Testing Data" + "       " + "Prediction" + "        " + "Probability" + "               " + "Review")
output_file.write ("Testing Data" + "       " + "Prediction" + "        " + "Probability" + "               " + "Review" + "\n")

for i in labels:
    folder = os.listdir(directory + i)
    for f in folder:
        path  = str(directory) + str(i) + "/" + str(f)
        filename = open(str(path), "r")
        for line in filename:
            line = line + " "
            si = 44
            if (len(line) >= si):
                while (line[si] != " "):
                    si += 1
                if (line[si-1].isalnum()):
                    line = line[0:si] + "..."
                else:
                    line = line[0:si-1] + "..."
            else:
                line = line[0:len(line)]
            break

        # Predict
        pred = nb.predict(str(path))

        # Print out some first results on screen; generate output file.
        if (i == "pos"):
            output_file.write ("  " + i + "               " + str(nb.pred_label_list[total]) + "               " + str(format(nb.pred_prob_list[total], '.4f')) + "           " + line + "\n")
            if (pos_index <= 23):
               print("  " + i + "               " + str(nb.pred_label_list[pos_index]) + "               " + str(format(nb.pred_prob_list[pos_index], '.4f')) + "           " + line)
            pos_index += 1
        else:
            if (neg_index == 0):
               print ()
               output_file.write ("\n")
            output_file.write ("  " + i + "               " + str(nb.pred_label_list[total]) + "               " + str(format(nb.pred_prob_list[total], '.4f')) + "           " + line + "\n")
            if (neg_index <= 23):
               print("  " + i + "               " + str(nb.pred_label_list[pos_index + neg_index]) + "               " + str(format(nb.pred_prob_list[pos_index + neg_index], '.4f')) + "           " + line)
            neg_index += 1

        if (i == str(nb.pred_label_list[total])):
            correct_pred += 1

        total += 1

print ()
output_file.write ("\n")

acc = (correct_pred / total)*100
print ("Accuracy: "+ str(round(acc, 2)) + "%.")
output_file.write ("Accuracy: "+ str(round(acc, 2)) + "%." + "\n")

end = time.time()
print ()
output_file.write ("\n")
print ("Execution time: " + str(round(end - start, 1)) + " seconds.")
output_file.write ("Execution time: " + str(round(end - start, 1)) + " seconds." + "\n")
print ()

output_file.close()
