
losses = {}
with open("randlog.txt", "r") as file:
    lines = file.readlines()

for line in lines:
    if "Loss" in line:
        losses[float(line[6:])] = lines.index(line)


sorted_keys = sorted(losses.keys())
for key in sorted_keys:
    print(losses[key])
