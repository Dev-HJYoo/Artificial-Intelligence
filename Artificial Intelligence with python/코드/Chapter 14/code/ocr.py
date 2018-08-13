import numpy as np
import neurolab as nl

# Define the input file
input_file = 'letter.data' # 입력 받기

# Define the number of datapoints to 
# be loaded from the input file
num_datapoints = 50

# String containing all the distinct characters 
orig_labels = 'omandig' # 학습시킬 단어

# Compute the number of distinct characters
num_orig_labels = len(orig_labels) # 단어의 길이

# Define the training and testing parameters  ## 이게잘 이해가 안되네 왜 결과가 5개가 나오는 거지...
num_train = int(0.9 * num_datapoints) # 학습  90%
num_test = num_datapoints - num_train # 예측  10%
 
# Define the dataset extraction parameters 
start = 6  # 파라미터 설정
end = -1

# Creating the dataset
data = [] # 데이터 셋 설정
labels = []
with open(input_file, 'r') as f:
    for line in f.readlines(): # 한 알파벳씩 가지고 오기
        # Split the current line tabwise
        list_vals = line.split('\t')

        # Check if the label is in our ground truth 
        # labels. If not, we should skip it.
        if list_vals[1] not in orig_labels: # 학습시킬 단어에 있는 알파벳이 아니면 통과
            continue

        # Extract the current label and append it 
        # to the main list
        label = np.zeros((num_orig_labels, 1)) # 라벨 모양으로 만들기
        label[orig_labels.index(list_vals[1])] = 1 # 학습시킬 단어에서 현재 알파벳의 위치에 1로 표현해서 라벨링
        labels.append(label) # 라벨 추가

        # Extract the character vector and append it to the main list
        cur_char = np.array([float(x) for x in list_vals[start:end]]) # 라벨링 된 데이터를 리스트에 추가
        data.append(cur_char)

        # Exit the loop once the required dataset has been created  
        if len(data) >= num_datapoints: # 50개가 학습 완료되면 탈출
            break

# Convert the data and labels to numpy arrays
data = np.asfarray(data) # float형태로 만들어줌 
labels = np.array(labels).reshape(num_datapoints, num_orig_labels) # 50개의 열과 7개의 행

# Extract the number of dimensions
num_dims = len(data[0]) # 한 줄의 데이터에서 입력의 크기

# Create a feedforward neural network
nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))],  # 신경망으로 네트워크 설정
        [128, 16, num_orig_labels])

# Set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd # 학습 준비

# Train the network
error_progress = nn.train(data[:num_train,:], labels[:num_train,:],  # 학습시키기 10000번 반복하고 100번만 보여준다. 목표는 0.01까지 줄이기
        epochs=10000, show=100, goal=0.01)

# Predict the output for test inputs 
print('\nTesting on unknown data:')
predicted_test = nn.sim(data[num_train:, :])
for i in range(num_test):
    print('\nOriginal:', orig_labels[np.argmax(labels[i])])
    print('Predicted:', orig_labels[np.argmax(predicted_test[i])])

