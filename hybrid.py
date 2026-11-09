POOL = 20

def fuse(bm_rank, sem_rank, w_bm=0.5, c=10):
    score = {}
    for ranking, w in ((bm_rank[:POOL], w_bm), (sem_rank[:POOL], 1.0)):
        for r, i in enumerate(ranking, start=1):
            score[i] = score.get(i, 0) + w / (c + r)
    return sorted(score, key=score.get, reverse=True)