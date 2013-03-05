
def average_precision(preds, refs, n=None):
    if n is None: n = len(refs)
    assert any(l in refs for l in preds)
    total = 0.0; num_correct = 0
    for k,pred in enumerate(preds):
        if pred in refs and pred != 'DUMMY':
            num_correct += 1
            p = float(num_correct/float(k+1))
            total += p
    return total / num_correct if total > 0 else 0.0


def one_error(preds, refs):
    return 0 if preds[0] in refs else 1

def is_error(ap):
	return 1 if ap < 1 else 0

def margin(preds, refs):
	lowest_relevant = 0
	highest_irrelevant = 0
	for k, pred in enumerate(preds):
		if pred not in refs and highest_irrelevant == 0:
			highest_irrelevant = k
		if pred in refs and pred > lowest_relevant:
			lowest_relevant = k
	return abs(lowest_relevant - highest_irrelevant)

	