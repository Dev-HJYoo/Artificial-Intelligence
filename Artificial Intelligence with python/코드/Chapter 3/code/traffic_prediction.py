import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn import cross_validation, preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import classification_report

# Load input data
input_file = 'traffic_data.txt' #요일, 시간, 길이름, 경기유무, 바이클 갯수
data = []
with open(input_file, 'r') as f: # input_file을 읽고 f 에 저장
    for line in f.readlines(): # 1줄씩 들고오고 -1빼고 저장
        items = line[:-1].split(',')
        data.append(items)

data = np.array(data)

# Convert string data to numerical data
# 숫자가 아닌 특징을 숫자형태로 인코딩하는 것.
label_encoder = [] 
X_encoded = np.empty(data.shape) # => data의 형태 처럼 만들기 print(data.shape) 해보기 => 아마도 (시간,경기유무) 일듯?

for i, item in enumerate(data[0]): # => enumerate는 i에는 인덱스 item에는 값이 들어간다. 
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

X = X_encoded[:, :-1].astype(int) # input으로 들어갈 특징들 
y = X_encoded[:, -1].astype(int) # 오토바이 갯수

# Split data into training and testing datasets 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.25, random_state=5)

# Extremely Random Forests regressor
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Compute the regressor performance on test data
y_pred = regressor.predict(X_test)
print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

# Testing encoding on single data instance
test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
test_datapoint = np.array(test_datapoint)
test_datapoint_encoded = [-1] * len(test_datapoint)

count = 0

for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        
        test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]])) ###### 여기에 [] 이거 하나 없다고 안돌아 갔다 왜냐!! 모양이 다르기 때문이지 [test_datapoint[i]] <- 요거 문제
        
        count = count + 1 

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Predict the output for the test datapoint
print("Predicted traffic:", int(regressor.predict([test_datapoint_encoded])[0]))

