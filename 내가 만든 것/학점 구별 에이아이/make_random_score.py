import random as ra

num_instance = 10000

f = open("C:/Users/BK/Desktop/test.txt", 'w') # 파일 이름 밑 선언 

for _ in range(num_instance):
    num = ra.randrange(50,101,2) # 랜덤으로 50~100에서 2단위로 가져온다
    clas = ""
    if num < 60: # 60 밑이면 F 
        clas = "F"
    elif num < 70:
        clas = "D"
    elif num < 80:
        clas = "C"
    elif num < 90:
        clas = "B"
    else :
        clas = "A"

    clas += "\n"
    f.write(str(num) + ",")
    f.write(clas)
f.close() # ex) 56,F 이렇게 저
    
        

