"""Command-line interface to rhasspyasr-deepspeech-hermes"""
import argparse
import asyncio
import logging
import typing
from pathlib import Path

import paho.mqtt.client as mqtt

import rhasspyhermes.cli as hermes_cli
from rhasspyasr_deepspeech import DeepSpeechTranscriber
from rhasspysilence import SilenceMethod

from . import AsrHermesMqtt

_LOGGER = logging.getLogger("rhasspyasr_deepspeech_hermes")

# -----------------------------------------------------------------------------


def main():
    """Main method."""
    args = get_args()

    hermes_cli.setup_logging(args)
    _LOGGER.debug(args)

    run_mqtt(args)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="rhasspy-asr-deepspeech-hermes")

    # Model settings
    parser.add_argument(
        "--model", required=True, help="Path to the model (protocol buffer binary file)"
    )
    parser.add_argument("--alphabet", help="Path to alphabet.txt file")
    parser.add_argument(
        "--language-model", help="Path to read/write ARPA language model file"
    )
    parser.add_argument(
        "--scorer",
        help="Path to the scorer package file created with native_client/generate_scorer_package",
    )
    parser.add_argument(
        "--beam-width", type=int, default=500, help="Beam width for the CTC decoder"
    )
    parser.add_argument(
        "--lm-alpha",
        type=float,
        default=0.931289039105002,
        help="Language model weight (lm_alpha)",
    )
    parser.add_argument(
        "--lm-beta",
        type=float,
        default=1.1834137581510284,
        help="Word insertion bonus (lm_beta)",
    )
    parser.add_argument(
        "--no-overwrite-train",
        action="store_true",
        help="Don't overwrite language model/scorer during training",
    )

    # Mixed language modeling
    parser.add_argument(
        "--base-language-model-fst",
        help="Path to base language model FST (training, mixed)",
    )
    parser.add_argument(
        "--base-language-model-weight",
        type=float,
        default=0,
        help="Weight to give base langauge model (training, mixed)",
    )
    parser.add_argument(
        "--mixed-language-model-fst",
        help="Path to write mixed langauge model FST (training, mixed)",
    )

    # Silence detection
    parser.add_argument(
        "--voice-skip-seconds",
        type=float,
        default=0.0,
        help="Seconds of audio to skip before a voice command",
    )
    parser.add_argument(
        "--voice-min-seconds",
        type=float,
        default=1.0,
        help="Minimum number of seconds for a voice command",
    )
    parser.add_argument(
        "--voice-max-seconds",
        type=float,
        help="Maximum number of seconds for a voice command",
    )
    parser.add_argument(
        "--voice-speech-seconds",
        type=float,
        default=0.3,
        help="Consecutive seconds of speech before start",
    )
    parser.add_argument(
        "--voice-silence-seconds",
        type=float,
        default=0.5,
        help="Consecutive seconds of silence before stop",
    )
    parser.add_argument(
        "--voice-before-seconds",
        type=float,
        default=0.5,
        help="Seconds to record before start",
    )
    parser.add_argument(
        "--voice-sensitivity",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="VAD sensitivity (1-3)",
    )
    parser.add_argument(
        "--voice-silence-method",
        choices=[e.value for e in SilenceMethod],
        default=SilenceMethod.VAD_ONLY,
        help="Method used to determine if an audio frame contains silence (see rhasspy-silence)",
    )
    parser.add_argument(
        "--voice-current-energy-threshold",
        type=float,
        help="Debiased energy threshold of current audio frame (see --voice-silence-method)",
    )
    parser.add_argument(
        "--voice-max-energy",
        type=float,
        help="Fixed maximum energy for ratio calculation (default: observed, see --voice-silence-method)",
    )
    parser.add_argument(
        "--voice-max-current-energy-ratio-threshold",
        type=float,
        help="Threshold of ratio between max energy and current audio frame (see --voice-silence-method)",
    )

    hermes_cli.add_hermes_args(parser)

    return parser.parse_args()


# -----------------------------------------------------------------------------


def run_mqtt(args: argparse.Namespace):
    """Runs Hermes ASR MQTT service."""
    # Convert to Paths
    args.model = Path(args.model)

    if args.language_model:
        args.language_model = Path(args.language_model)

    if args.scorer:
        args.scorer = Path(args.scorer)

    if args.alphabet:
        args.alphabet = Path(args.alphabet)

    if args.base_language_model_fst:
        args.base_language_model_fst = Path(args.base_language_model_fst)

    if args.mixed_language_model_fst:
        args.mixed_language_model_fst = Path(args.mixed_language_model_fst)

    def make_transcriber():
        return DeepSpeechTranscriber(
            args.model,
            args.scorer,
            beam_width=args.beam_width,
            lm_alpha=args.lm_alpha,
            lm_beta=args.lm_beta,
        )

    # Listen for messages
    client = mqtt.Client()
    hermes = AsrHermesMqtt(
        client,
        transcriber_factory=make_transcriber,
        language_model_path=args.language_model,
        scorer_path=args.scorer,
        alphabet_path=args.alphabet,
        base_language_model_fst=args.base_language_model_fst,
        base_language_model_weight=args.base_language_model_weight,
        mixed_language_model_fst=args.mixed_language_model_fst,
        no_overwrite_train=args.no_overwrite_train,
        skip_seconds=args.voice_skip_seconds,
        min_seconds=args.voice_min_seconds,
        max_seconds=args.voice_max_seconds,
        speech_seconds=args.voice_speech_seconds,
        silence_seconds=args.voice_silence_seconds,
        before_seconds=args.voice_before_seconds,
        vad_mode=args.voice_sensitivity,
        silence_method=args.voice_silence_method,
        current_energy_threshold=args.voice_current_energy_threshold,
        max_energy=args.voice_max_energy,
        max_current_energy_ratio_threshold=args.voice_max_current_energy_ratio_threshold,
        site_ids=args.site_id,
    )

    _LOGGER.debug("Connecting to %s:%s", args.host, args.port)
    hermes_cli.connect(client, args)
    client.loop_start()

    try:
        # Run event loop
        asyncio.run(hermes.handle_messages_async())
    except KeyboardInterrupt:
        pass
    finally:
        _LOGGER.debug("Shutting down")
        client.loop_stop()


# -----------------------------------------------------------------------------


def get_word_transform(name: str) -> typing.Optional[typing.Callable[[str], str]]:
    """Gets a word transformation function by name."""
    if name == "upper":
        return str.upper

    if name == "lower":
        return str.lower

    return None


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
