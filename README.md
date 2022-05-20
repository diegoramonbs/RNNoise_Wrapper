# RNNoise Wrapper


## Install using pip:
```bash
pip install git+https://github.com/Desklop/RNNoise_Wrapper
```

## Install dependencies
```bash
sudo apt-get install autoconf libtool
```

## Build RNNnoise 

```bash
git clone https://github.com/Desklop/RNNoise_Wrapper
cd RNNoise_Wrapper
./compile_rnnoise.sh
```
## Basic usage

```python
from rnnoise_wrapper import RNNoise

denoiser = RNNoise()

audio = denoiser.read_wav('test.wav')
denoised_audio = denoiser.filter(audio)
denoiser.write_wav('test_denoised.wav', denoised_audio)
```


## Acknowledgments


This code is a cleaned up version of this code [RNNoise_Wrapper](https://github.com/Desklop/RNNoise_Wrapper)


---

## Contributors

Vladislav Klim - vladsklim@gmail.com или в [LinkedIn](https://www.linkedin.com/in/vladklim/).
