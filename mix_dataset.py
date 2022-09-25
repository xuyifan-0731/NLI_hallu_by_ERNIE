

with open("data/Hallu2/train/train") as f1,open("data/chineseNLI/train/train") as f2,open("data/mixdataset_75NLI_25conv/train/train","w", encoding='utf-8') as f3:
    num = 0
    for line1 in f1:
        f3.write(line1)

        line2 = f2.readline()
        line2 = line2.replace("contradiction","contradictory")
        f3.write(line2)
        line2 = f2.readline()
        line2 = line2.replace("contradiction","contradictory")
        f3.write(line2)
        line2 = f2.readline()
        line2 = line2.replace("contradiction","contradictory")
        f3.write(line2)

        num = num + 1
        if num==30000:
            break
