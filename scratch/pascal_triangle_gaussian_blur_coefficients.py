values = [1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1]
s = sum(values)
weighted = []
for v in values:
    weighted.append(v / s)

for w in weighted:
    print('{0:.10f}'.format(w))

# [0.2094726562, 0.1832885742, 0.1221923828, 0.0610961914, 0.0222167969, 0.0055541992, 0.0008544922, 0.0000610352]