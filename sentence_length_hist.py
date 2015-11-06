import re
import matplotlib.pyplot as plt

lens = []
prev = 0
sents = 0
with open('amr.txt') as f:
    for line in f:
        if re.search("(PROXY_AFP_ENG.*\.txt)", line) is None:
            continue

        curr = int(re.search("(PROXY_AFP_ENG.*\.txt)", line).group(1).split("_")[-1].split(".")[0])
        if curr < prev:
            lens.append(prev) 
            sents += prev
        prev = curr

print(len(lens))
print(lens)
print(sents)
plt.hist(lens)
plt.show()