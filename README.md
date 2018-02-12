# Multi-Speaker Tacotron in TensorFlow

원본소스를 따라해보고 이해한대로 한국말로 번역하여 작성함.
[원본소스](https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow)

관련 논문:

- [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947)
- [Listening while Speaking: Speech Chain by Deep Learning](https://arxiv.org/abs/1707.04879)
- [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)

원본소스 공개자의 샘플 페이지 [here](http://carpedm20.github.io/tacotron/en.html).

## 필수요소

- Python 3.6+(3.6이상이라고 했는데 3.7로 하면 소스코드를 수정 해야함)
- FFmpeg
- [Tensorflow 1.3](https://www.tensorflow.org/install/)


## 사용법

### 1. 환경 설정

#### 1-1. 아나콘다 설치
편한 설치환셩을 위해 아나콘다를 사용했다. [참고](https://anaconda.org/)

아나콘다 프롬프트에서 가상 환경을 만든다.
    conda create -n tacotron python=3.5

가상환경을 활성화 한다.(OS 환경에 따른 명령어)
    (source)activate tacotron

#### 1-2. 텐서 플로우 설치
텐서플로우를 설치한다. [참고](https://www.tensorflow.org/install/)
1.3.0버전을 whl을 다운받아 설치했다.[다운로드](https://pypi.python.org/pypi/tensorflow/1.3.0)
    pip install tensorflow-1.3.0-$PYTHONVER-$PYTHONVER-$OS-$BIT.Whl(다운로드 받은 파일명 적으면 됨. 파이선 버전이랑 OS/OS환경에 주의하여 다운로드)

#### 1-3. ffmpeg 설치
FFmpeg를 설치한다.
    pip install ffmpeg-normalize

####
필수 요소 설치 명령어 :

    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt')"


### 2-1. 데이터셋 만들기
다른 데이터셋 만들때도 필요하므로 일단 둠

데이터셋 디렉토리는 다음과 같이 만들어져야 함 : 

    datasets
    ├── son
    │   ├── alignment.json
    │   └── audio
    │       ├── 1.mp3
    │       ├── 2.mp3
    │       ├── 3.mp3
    │       └── ...
    └── YOUR_DATASET
        ├── alignment.json
        └── audio
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            └── ...

`YOUR_DATASET/alignment.json` 파일의 내용은 아래와 같이 작성되어야 함.

    {
        "./datasets/YOUR_DATASET/audio/001.mp3": "My name is Taehoon Kim.",
        "./datasets/YOUR_DATASET/audio/002.mp3": "The buses aren't the problem.",
        "./datasets/YOUR_DATASET/audio/003.mp3": "They have discovered a new particle.",
    }

데이터 변형하는 명령어 : 

    python -m datasets.generate_data ./datasets/YOUR_DATASET/alignment.json


### 2-2. 한국어 데이터셋

아래 명령어를 따라하면 손석희의 데이터셋을 받을 수 있음.

0. 음성을 자동으로 텍스트로 변환하기 위해  [Google Speech Recognition API](https://cloud.google.com/speech/) 을 사용한다. 인증 JSON을 받으면 됨 [참고](https://developers.google.com/identity/protocols/application-default-credentials).

       export GOOGLE_APPLICATION_CREDENTIALS="YOUR-GOOGLE.CREDENTIALS.json"

1. 손석희 데이터 다운로드 스크립트 실행

       python3 -m datasets.son.download

2. 음성파일들이 매우 길어서 이걸 문장별로 자른다.

       python3 -m audio.silence --audio_pattern "./datasets/son/audio/*.wav" --method=pydub

3. [Google Speech Recognition API](https://cloud.google.com/speech/) 을 이용하여 문장별로 자른 음성파일들에 대한 텍스트 파일을 만든다. 
주의 : 이 스크립트 파일은 한번에 모든 음성파일을 실행하고 에러가 5번 나면 멈추게 되어있다.
내 코드는 api에러와 상관없이 처음부터 끝까지 모두 api요청하는데 해당 요청을 한번에 모두 하면 200$요금이 나옵니다.

       python3 -m recognition.google --audio_pattern "./datasets/son/audio/*.*.wav"

4. 데이터를 조정하는단계 같은데 뭔지 잘 모르겠고 [이 이슈](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/issues/4)에 잘못된 데이터도 많고 해서 이 단계는 별 의미 없는 것으로 판단하고 skip함

       python3 -m recognition.alignment --recognition_path "./datasets/son/recognition.json" --score_threshold=0.5

5. 이걸 트레이닝에 쓰일 각각의 numpy 파일로 생성한다.

       python3 -m datasets.generate_data ./datasets/son/alignment.json

Because the automatic generation is extremely naive, the dataset is noisy. However, if you have enough datasets (20+ hours with random initialization or 5+ hours with pretrained model initialization), you can expect an acceptable quality of audio synthesis.

설정을 몇번 진행하다 우분투에서 진행하던 중 [이 문제](https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/issues/10) 와 같은 현상이 있었는데 ffmpeg 설치 -> generate_data 재실행 -> plot.py 에 명시적인 폰트 경로/이름 등 설정으로 해결함

### 2-3. 영어 데이터 변환(이 단계는 skip )

1. Download speech dataset at https://keithito.com/LJ-Speech-Dataset/

2. Convert metadata CSV file to json file. (arguments are available for changing preferences)
		
		python3 -m datasets.LJSpeech_1_0.prepare

3. Finally, generate numpy files which will be used in training.
		
		python3 -m datasets.generate_data ./datasets/LJSpeech_1_0
		

### 3. 모델 훈련

중요 파라미터는 `hparams.py` 파일에 정의 되어 있음

싱글 스피커 모델 트레인:

    python3 train.py --data_path=datasets/son
    python3 train.py --data_path=datasets/son --initialize_path=PATH_TO_CHECKPOINT

멀티 스피커 모델 트레인:

    # after change `model_type` in `hparams.py` to `deepvoice` or `simple`
    python3 train.py --data_path=datasets/son1,datasets/son2

다시 트레인 할때 이전 생성한 모델을 이용하는 방법:

    python3 train.py --data_path=datasets/son --load_path logs/son-20171015

	
## 아직 트레인 하는 중..
If you don't have good and enough (10+ hours) dataset, it would be better to use `--initialize_path` to use a well-trained model as initial parameters.


### 4. Synthesize audio

You can train your own models with:

    python3 app.py --load_path logs/son-20171015 --num_speakers=1

or generate audio directly with:

    python3 synthesizer.py --load_path logs/son-20171015 --text "이거 실화냐?"
	
### 4-1. Synthesizing non-korean(english) audio

For generating non-korean audio, you must set the argument --is_korean False.
		
	python3 app.py --load_path logs/LJSpeech_1_0-20180108 --num_speakers=1 --is_korean=False
	python3 synthesizer.py --load_path logs/LJSpeech_1_0-20180108 --text="Winter is coming." --is_korean=False

## Results

Training attention on single speaker model:

![model](./assets/attention_single_speaker.gif)

Training attention on multi speaker model:

![model](./assets/attention_multi_speaker.gif)


## Disclaimer

This is not an official [DEVSISTERS](http://devsisters.com/) product. This project is not responsible for misuse or for any damage that you may cause. You agree that you use this software at your own risk.


## References

- [Keith Ito](https://github.com/keithito)'s [tacotron](https://github.com/keithito/tacotron)
- [DEVIEW 2017 presentation](https://www.slideshare.net/carpedm20/deview-2017-80824162)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
