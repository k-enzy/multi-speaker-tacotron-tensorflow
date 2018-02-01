# Multi-Speaker Tacotron in TensorFlow

원본소스를 따라해보고 이해한대로 한국말로 번역하여 작성함.
[원본소스](https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow)

관련 논문:

- [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947)
- [Listening while Speaking: Speech Chain by Deep Learning](https://arxiv.org/abs/1707.04879)
- [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)

원본소스 공개자의 샘플 페이지 [here](http://carpedm20.github.io/tacotron/en.html).

![model](./assets/model.png)


## 필수요소

- Python 3.6+(3.6이상이라고 했는데 3.7로 하면 소스코드를 수정 해야함)
- FFmpeg
- [Tensorflow 1.3](https://www.tensorflow.org/install/)


## 사용법

### 1. 설치

텐서플로우를 설치한다. [참고](https://www.tensorflow.org/install/)
필수 요소 설치 명령어 :

    pip3 install -r requirements.txt
    python -c "import nltk; nltk.download('punkt')"


### 2-1. 데이터셋 만들기

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

    python3 -m datasets.generate_data ./datasets/YOUR_DATASET/alignment.json


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

4. By comparing original text and recognised text, save `audio<->text` pair information into `./datasets/son/alignment.json`.

       python3 -m recognition.alignment --recognition_path "./datasets/son/recognition.json" --score_threshold=0.5

5. Finally, generated numpy files which will be used in training.

       python3 -m datasets.generate_data ./datasets/son/alignment.json

Because the automatic generation is extremely naive, the dataset is noisy. However, if you have enough datasets (20+ hours with random initialization or 5+ hours with pretrained model initialization), you can expect an acceptable quality of audio synthesis.

### 2-3. Generate English datasets

1. Download speech dataset at https://keithito.com/LJ-Speech-Dataset/

2. Convert metadata CSV file to json file. (arguments are available for changing preferences)
		
		python3 -m datasets.LJSpeech_1_0.prepare

3. Finally, generate numpy files which will be used in training.
		
		python3 -m datasets.generate_data ./datasets/LJSpeech_1_0
		

### 3. Train a model

The important hyperparameters for a models are defined in `hparams.py`.

(**Change `cleaners` in `hparams.py` from `korean_cleaners` to `english_cleaners` to train with English dataset**)

To train a single-speaker model:

    python3 train.py --data_path=datasets/son
    python3 train.py --data_path=datasets/son --initialize_path=PATH_TO_CHECKPOINT

To train a multi-speaker model:

    # after change `model_type` in `hparams.py` to `deepvoice` or `simple`
    python3 train.py --data_path=datasets/son1,datasets/son2

To restart a training from previous experiments such as `logs/son-20171015`:

    python3 train.py --data_path=datasets/son --load_path logs/son-20171015

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
