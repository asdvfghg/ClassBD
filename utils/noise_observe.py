import numpy as np

def add_audio_noise(noise, data):
    m, n = data.shape
    noise = noise[:,0]
    noise_start = np.argwhere(noise != 0.0)
    noise = noise[noise_start]
    Noise_sample = []
    for i in range(m):
        random_start = np.random.randint(low=0, high=len(noise)-n)
        Noise_sample.append(noise[random_start:random_start + n])
    Noise = np.array(Noise_sample).squeeze() * 100
    return data + Noise



if __name__ == '__main__':
    noise1  = np.load('airplanenoise1e-10.npy')
    noise2  = np.load('pinknoise_1e-10.npy')
    noise3  = np.load('trucknoise1e-10.npy')
    noise1 = noise1[:, 0]
    noise2 = noise2[:, 0]
    noise3 = noise3[:, 0]
    noise_start = np.argwhere(noise1!=0.0)
    noise1 = noise1[noise_start]
    print('a')