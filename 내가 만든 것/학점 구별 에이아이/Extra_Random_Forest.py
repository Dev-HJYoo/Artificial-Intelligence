import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utilities import visualize_classifier

if __name__ == '__main__':

    input_file = "test.txt" # test 파일을 불러온다 
    f = open(input_file, 'r')
    encoder_data = [ 'A', 'B', 'C', 'D', 'F'] # 학점 라벨링
    encoder = preprocessing.LabelEncoder() # 인코딩 1->A 2->B 이런형태로
    encoder.fit(encoder_data) # 학습
    datas = []
    while True:
        line = f.readline() # 한줄 읽어 오기
        if not line: # 끝나기 전까지 실행
            break 
        s = (line.strip()).split(',') # ,를 기준으로 나누기
        X = int(s[0]) # 점수 
        y = int(encoder.transform([s[1]])) # 학점 라벨링 ( 숫자로 만듬)
        data = [X]
        data += [y] # 리스트 형태로 저장
        datas.append(data) #전체 저장

    data = np.array(datas) # np형태로 만들기
    X, y = data[:, :-1], data[:, -1] # 입력과 라벨로 나누기
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split( # 25%를 시험으로 사용 
        X, y, test_size = 0.25, random_state = 5)

    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0} # ERF 파라미터
    
    classifier = ExtraTreesClassifier(**params) # ERF

    classifier.fit(X_train, y_train) # 학습
    fs = open("end.txt", 'w') # 저장 파일 

    i = 0
    for test in X_test: # 예측 값을 저장
        check = ""
        probabilities = classifier.predict_proba([test])[0]
        print(probabilities)
        max_pro = max(probabilities)
        predict = list(probabilities).index(max_pro)
        if predict == y_test[i]: # 예측값과 실제 값과 맞는지 체크
            check = "True"
        
        i += 1
        fs.write(str(test[:]) + " => " + encoder.inverse_transform(predict) + " "+  check + " \n")

    f.close()
    fs.close()
    
        
    
