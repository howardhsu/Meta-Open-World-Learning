

avgs=[]
for baseline in ["l2ac_1_10", "l2ac_3_10", "l2ac_5_10", "l2ac_10_10", "l2ac_15_10", "l2ac_20_10"]:
    scores=np.zeros((runs, 6) )
    for run in range(runs):
        run_dir="../runs/"+baseline+"/"+str(run)+"/"
        with open(run_dir+"eval.json") as f:
            score=json.load(f)
        scores[run, 0], scores[run, 1], scores[run, 2], scores[run, 3], scores[run, 4], scores[run, 5]=score['test_25']['weighted_f1'], score['test_25']['macro_f1'], score['test_50']['weighted_f1'], score['test_50']['macro_f1'], score['test_75']['weighted_f1'], score['test_75']['macro_f1']
    scores*=100
    avg, std=np.average(scores, axis=0), np.std(scores, axis=0)
    avgs.append(avg)
    print name[baseline], "&", " & ".join([str(round(avg[ix], 2) )+"("+str(round(std[ix], 2) )+")" for ix in range(6)] ),"\\\\"
avgs=np.vstack(avgs)

