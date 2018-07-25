import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

def get_data(num_points):
    # Create sine waveforms
    wave_1 = 0.5 * np.sin(np.arange(0, num_points)) # 사인 그래프 모양으로 만듬
    wave_2 = 3.6 * np.sin(np.arange(0, num_points))
    wave_3 = 1.1 * np.sin(np.arange(0, num_points))
    wave_4 = 4.7 * np.sin(np.arange(0, num_points))

    # Create varying amplitudes
    amp_1 = np.ones(num_points) # 1로 초기화된 numpy 배열 생성
    amp_2 = 2.1 + np.zeros(num_points) # 0으로 초기화된 numpy 배열 생
    amp_3 = 3.2 * np.ones(num_points) 
    amp_4 = 0.8 + np.zeros(num_points) 

    wave = np.array([wave_1, wave_2, wave_3, wave_4]).reshape(num_points * 4, 1) # 사인 모양
    amp = np.array([[amp_1, amp_2, amp_3, amp_4]]).reshape(num_points * 4, 1) # 진폭

    return wave, amp 

# Visualize the output 
def visualize_output(nn, num_points_test):
    wave, amp = get_data(num_points_test)
    output = nn.sim(wave)
    plt.plot(amp.reshape(num_points_test * 4))
    plt.plot(output.reshape(num_points_test * 4))

if __name__=='__main__':
    # Create some sample data
    num_points = 40
    wave, amp = get_data(num_points) # 데이터 생성

    # Create a recurrent neural network with 2 layers
    nn = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()]) # 2개의 층으로 이루어진 rnn 네트워크 생성
    print(nn)
    # Set the init functions for each layer 
    nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb') 
    nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.init() # 초기화 시키는 것 or 생성자

    # Train the recurrent neural network
    error_progress = nn.train(wave, amp, epochs=1200, show=100, goal=0.01) # 학습

    # Run the training data through the network
    output = nn.sim(wave) # 예측값 

    # Plot the results
    plt.subplot(211)
    plt.plot(error_progress) # 에러 정도
    plt.xlabel('Number of epochs')
    plt.ylabel('Error (MSE)')

    plt.subplot(212)
    plt.plot(amp.reshape(num_points * 4)) # 예측 값과 실제 값 비교
    plt.plot(output.reshape(num_points * 4))
    plt.legend(['Original', 'Predicted'])

    # Testing the network performance on unknown data 
    plt.figure() # 임의의 데이터를 넣었을때 비교 

    plt.subplot(211)
    visualize_output(nn, 82)
    plt.xlim([0, 300])

    plt.subplot(212)
    visualize_output(nn, 49)
    plt.xlim([0, 300])

    plt.show()
