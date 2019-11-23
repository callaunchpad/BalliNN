from pipeline import Model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
m = Model(reg)

m.train([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(m.get_coefs())
print(m.get_intercept())
print(m.getScore( [[4, 6], [3, 7]], [0, 0] ))