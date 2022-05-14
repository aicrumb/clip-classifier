from fastsr.estimators.symbolic_regression import SymbolicRegression
import numpy as np

def quick_solve(x, y, x_test=None, y_test=None, ngen=100, pop_size=100, verbose=False):
	"""
		this is just because im really terrible at simple math and need something to reason for me

		args
		x: numpy ndarray (n_samples, n_datapoints)
		y: numpy ndarray (n_samples, 1)
		x_test: same as x
		y_test: same as y
		ngen: integer, 1-inf
		pop_size: integer, 1-inf
		verbose: boolean
	"""
	sr = SymbolicRegression(ngen=ngen, pop_size=pop_size)
	sr.fit(x, y)
	if x_test and y_test:
		score = sr.score(x_test, y_test)
	else:
		score = sr.score(x, y)
	if verbose:
		print('Score: {}'.format(score))
		print('Best Individuals:')
		sr.print_best_individuals()
	return [str(ind) for ind in sr.best_individuals_], [ind.error for in in sr.best_individuals_]