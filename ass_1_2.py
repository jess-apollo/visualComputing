import numpy as np
import matplotlib.pyplot as plt

def print_mean_cov(mf, cov, class_num) :
    print "Sepal length mean : %f" % mf[class_num-1][0]
    print "Sepal width mean : %f" % mf[class_num-1][1]
    
    print "-Covariance-"
    print "  sepal_legth  sepal_width"
    print cov
    print ("\n")

train_dat_file = open("Iris_train.dat", "r")

feature = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]

for line in train_dat_file :
    data = [float(x) for x in line.rstrip().split()]
    
    if(data[4] == 1.0) :
        feature[0][0].append(data[0])
        feature[0][1].append(data[1])
        feature[0][2].append(data[2])
        feature[0][3].append(data[3])
    elif(data[4] == 2.0) :
        feature[1][0].append(data[0])
        feature[1][1].append(data[1])
        feature[1][2].append(data[2])
        feature[1][3].append(data[3])
    elif(data[4] == 3.0) :
        feature[2][0].append(data[0])
        feature[2][1].append(data[1])
        feature[2][2].append(data[2])
        feature[2][3].append(data[3])
    
print "*2 features : Sepal Length, Sepal width"
# Iris Setosa
plt.plot(feature[0][0], feature[0][1], 'ro')
plt.plot(np.mean(feature[0][0]), np.mean(feature[0][1]), 'co')
# Iris Versicolor
plt.plot(feature[1][0], feature[1][1], 'bs')
plt.plot(np.mean(feature[1][0]), np.mean(feature[1][1]), 'cs')
# Iris Virginica
plt.plot(feature[2][0], feature[2][1], 'y^')
plt.plot(np.mean(feature[2][0]), np.mean(feature[2][1]), 'c^')

for i in range(3) :
    for j in range(4) :
        mean_feature[i].append(np.mean(feature[i][j]))

cov_feature1 = np.cov(feature[0][0], feature[0][1])
cov_feature2 = np.cov(feature[1][0], feature[1][1])
cov_feature3 = np.cov(feature[2][0], feature[2][1])

print "<Iris Setosa>"       
print_mean_cov(mean_feature, cov_feature1, 1)

print "<Iris Versicolor>"       
print_mean_cov(mean_feature, cov_feature1, 2)

print "<Iris Virginica>"       
print_mean_cov(mean_feature, cov_feature1, 3)

# equation
delta = 0.01
x = np.arange(3.5, 9.0, delta)
y = np.arange(1.5, 5.0, delta)
X, Y = np.meshgrid(x, y)

acov1 = np.linalg.inv(cov_feature1)
acov2 = np.linalg.inv(cov_feature2)
acov3 = np.linalg.inv(cov_feature3)

mahalanobis_d1 = np.sqrt(acov1[0][0]*np.power((X-mean_feature[0][0]),2) + (acov1[0][1] + acov1[1][0])*(Y-mean_feature[0][1])*(X-mean_feature[0][0]) + acov1[1][1]*np.power((Y-mean_feature[0][1]),2)) - 2 
plt.contour(X, Y, mahalanobis_d1, [0], colors = 'red')
mahalanobis_d2 = np.sqrt(acov2[0][0]*np.power((X-mean_feature[1][0]),2) + (acov2[0][1] + acov2[1][0])*(Y-mean_feature[1][1])*(X-mean_feature[1][0]) + acov2[1][1]*np.power((Y-mean_feature[1][1]),2)) - 2 
plt.contour(X, Y, mahalanobis_d2, [0], colors = 'blue')
mahalanobis_d3 = np.sqrt(acov3[0][0]*np.power((X-mean_feature[2][0]),2) + (acov3[0][1] + acov3[1][0])*(Y-mean_feature[2][1])*(X-mean_feature[2][0]) + acov3[1][1]*np.power((Y-mean_feature[2][1]),2)) - 2 
plt.contour(X, Y, mahalanobis_d3, [0], colors = 'yellow')


decision_boundary1 = mahalanobis_d1 - mahalanobis_d2
decision_boundary2 = mahalanobis_d2 - mahalanobis_d3
decision_boundary3 = mahalanobis_d3 - mahalanobis_d1

plt.contour(X, Y, decision_boundary1, [0], colors = 'black')
plt.contour(X, Y, decision_boundary2, [0], colors = 'green')
plt.contour(X, Y, decision_boundary3, [0], colors = 'magenta')


# << x >>
test_dat_file = open("Iris_test.dat", "r")

test_feature = [[], [], [], [], []]

for line in test_dat_file :
    data = [float(x) for x in line.rstrip().split()]
    
    test_feature[0].append(data[0])
    test_feature[1].append(data[1])
    test_feature[2].append(data[2])
    test_feature[3].append(data[3])
    test_feature[4].append(data[4])

plt.show()
