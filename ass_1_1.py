import numpy as np
from numpy.linalg import inv

def print_mean_cov(mf, cov, class_num) :
    print "Sepal length mean : %f" % mf[class_num-1][0]
    print "Sepal width mean : %f" % mf[class_num-1][1]
    print "Petal length mean : %f" % mf[class_num-1][2]
    print "Petal width mean : %f \n"  % mf[class_num-1][3]
    
    print "-Covariance-"
    print "  sepal_legth  sepal_width petal_legth petal_width"
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
        
mean_feature = [[], [], []]

for i in range(3) :
    for j in range(4) :
        mean_feature[i].append(np.mean(feature[i][j]))

cov_feature1 = np.cov([feature[0][0], feature[0][1], feature[0][2], feature[0][3]])
cov_feature2 = np.cov([feature[1][0], feature[1][1], feature[1][2], feature[1][3]])
cov_feature3 = np.cov([feature[2][0], feature[2][1], feature[2][2], feature[2][3]])

print "<Iris Setosa>"       
print_mean_cov(mean_feature, cov_feature1, 1)

print "<Iris Versicolor>"       
print_mean_cov(mean_feature, cov_feature1, 2)

print "<Iris Virginica>"       
print_mean_cov(mean_feature, cov_feature1, 3)


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

# << Vi >>
Vi1 = -0.5 * np.linalg.inv(cov_feature1)
Vi2 = -0.5 * np.linalg.inv(cov_feature2)
Vi3 = -0.5 * np.linalg.inv(cov_feature3)

# << vi_m >>
vi_m1 = np.dot(np.linalg.inv(cov_feature1), mean_feature[0])
vi_m2 = np.dot(np.linalg.inv(cov_feature2), mean_feature[1])
vi_m3 = np.dot(np.linalg.inv(cov_feature3), mean_feature[2])

# << vi0 >>
vi01 = (-0.5) * np.dot(np.transpose(mean_feature[0]), vi_m1) + (-0.5)*np.log(np.linalg.det(cov_feature1)) + np.log(0.3)
vi02 = (-0.5) * np.dot(np.transpose(mean_feature[1]), vi_m2) + (-0.5)*np.log(np.linalg.det(cov_feature2)) + np.log(0.3)
vi03 = (-0.5) * np.dot(np.transpose(mean_feature[2]), vi_m3) + (-0.5)*np.log(np.linalg.det(cov_feature3)) + np.log(0.3)

confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]

for i in range(len(test_feature[0])):
    test = [test_feature[0][i], test_feature[1][i], test_feature[2][i], test_feature[3][i]]

    g1 = np.dot(np.dot(np.transpose(test), Vi1), test) + np.dot(np.transpose(vi_m1), test) + vi01
    g2 = np.dot(np.dot(np.transpose(test), Vi2), test) + np.dot(np.transpose(vi_m2), test) + vi02
    g3 = np.dot(np.dot(np.transpose(test), Vi3), test) + np.dot(np.transpose(vi_m3), test) + vi03

    maxG = max([g1, g2, g3])
    
    if(maxG == g1 and test_feature[4][i] == 1):
        confusion_matrix[0][0] += 1
    elif(maxG == g1 and test_feature[4][i] == 2):
        confusion_matrix[0][1] += 1
    elif(maxG == g1 and test_feature[4][i] == 3):
        confusion_matrix[0][2] += 1
    elif(maxG == g2 and test_feature[4][i] == 1):
        confusion_matrix[1][0] += 1
    elif(maxG == g2 and test_feature[4][i] == 2):
        confusion_matrix[1][1] += 1
    elif(maxG == g2 and test_feature[4][i] == 3):
        confusion_matrix[1][2] += 1
    elif(maxG == g3 and test_feature[4][i] == 1):
        confusion_matrix[2][0] += 1
    elif(maxG == g3 and test_feature[4][i] == 2):
        confusion_matrix[2][1] += 1
    elif(maxG == g3 and test_feature[4][i] == 3):
        confusion_matrix[2][2] += 1

print("Confusion Matrix")
print(np.transpose(confusion_matrix))

