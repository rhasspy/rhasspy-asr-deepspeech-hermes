# Rhasspy ASR DeepSpeech Hermes

MQTT service for Rhasspy that uses Mozilla's DeepSpeech 0.9.3.

## Requirements

* Python 3.7
* [Mozilla DeepSpeech 0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3)
* [Pre-trained model](https://github.com/mozilla/DeepSpeech/blob/master/doc/USING.rst#getting-the-pre-trained-model)
* `generate_scorer_package` in `PATH` from [native client](https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cpu.linux.tar.xz)
* `build_binary` in `PATH` from [KenLM](https://github.com/kpu/kenlm)

## Installing

Clone the repository and create a virtual environment:

```bash
$ git clone https://github.com/rhasspy/rhasspy-asr-deepspeech-hermes.git
$ cd rhasspy-asr-deepspeech-hermes
$ ./configure
$ make
$ make install
```

## Running

Run script:

```bash
bin/rhasspy-asr-deepspeech-hermes \
    --model /path/to/output_graph.pbmm \
    --language-model /path/to/lm.binary \
    --trie /path/to/trie \
    --host <MQTT_HOST> \
    --port <MQTT_PORT> \
    --debug
```

## Using

Set Rhasspy ASR system to "Hermes MQTT". Connect Rhasspy and DeepSpeech service to the same MQTT broker (use port 12183 for Rhasspy's internal broker).
