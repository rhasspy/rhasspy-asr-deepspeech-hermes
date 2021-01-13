"""Hermes MQTT server for Rhasspy ASR using Mozilla's DeepSpeech"""
import gzip
import logging
import threading
import typing
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

import networkx as nx
from rhasspyasr import Transcriber, Transcription

import rhasspyasr_deepspeech
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrRecordingFinished,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrToggleReason,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspysilence import SilenceMethod, VoiceCommandRecorder, WebRtcVadRecorder

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("rhasspyasr_deepspeech_hermes")

# -----------------------------------------------------------------------------

AudioCapturedType = typing.Tuple[AsrAudioCaptured, TopicArgs]
StopListeningType = typing.Union[
    AsrRecordingFinished, AsrTextCaptured, AsrError, AudioCapturedType
]


@dataclass
class TranscriberInfo:
    """Objects for a single transcriber"""

    transcriber: typing.Optional[Transcriber] = None
    recorder: typing.Optional[VoiceCommandRecorder] = None
    frame_queue: "Queue[typing.Optional[bytes]]" = field(default_factory=Queue)
    ready_event: threading.Event = field(default_factory=threading.Event)
    result: typing.Optional[Transcription] = None
    result_event: threading.Event = field(default_factory=threading.Event)
    result_sent: bool = False
    start_listening: typing.Optional[AsrStartListening] = None
    thread: typing.Optional[threading.Thread] = None
    audio_buffer: typing.Optional[bytes] = None
    reuse: bool = True


# -----------------------------------------------------------------------------


class AsrHermesMqtt(HermesClient):
    """Hermes MQTT server for Rhasspy ASR using Mozilla's DeepSpeech."""

    def __init__(
        self,
        client,
        transcriber_factory: typing.Callable[[], Transcriber],
        language_model_path: typing.Optional[Path] = None,
        alphabet_path: typing.Optional[Path] = None,
        scorer_path: typing.Optional[Path] = None,
        no_overwrite_train: bool = False,
        base_language_model_fst: typing.Optional[Path] = None,
        base_language_model_weight: float = 0,
        mixed_language_model_fst: typing.Optional[Path] = None,
        site_ids: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        recorder_factory: typing.Callable[[], VoiceCommandRecorder] = None,
        skip_seconds: float = 0.0,
        min_seconds: float = 1.0,
        max_seconds: typing.Optional[float] = None,
        speech_seconds: float = 0.3,
        silence_seconds: float = 0.5,
        before_seconds: float = 0.5,
        vad_mode: int = 3,
        max_energy: typing.Optional[float] = None,
        max_current_energy_ratio_threshold: typing.Optional[float] = None,
        current_energy_threshold: typing.Optional[float] = None,
        silence_method: SilenceMethod = SilenceMethod.VAD_ONLY,
        session_result_timeout: float = 20,
        reuse_transcribers: bool = True,
    ):
        super().__init__(
            "rhasspyasr_deepspeech_hermes",
            client,
            site_ids=site_ids,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        self.subscribe(
            AsrToggleOn,
            AsrToggleOff,
            AsrStartListening,
            AsrStopListening,
            AudioFrame,
            AudioSessionFrame,
            AsrTrain,
        )

        self.transcriber_factory = transcriber_factory

        # Mixed language model
        self.base_language_model_fst = base_language_model_fst
        self.base_language_model_weight = base_language_model_weight
        self.mixed_language_model_fst = mixed_language_model_fst

        # If True, language model/scorer won't be overwritten during training
        self.no_overwrite_train = no_overwrite_train

        # True if transcribers should be reused
        self.reuse_transcribers = reuse_transcribers

        # Files to write during training
        self.language_model_path = language_model_path
        self.scorer_path = scorer_path
        self.alphabet_path = alphabet_path

        # True if ASR system is enabled
        self.enabled = enabled
        self.disabled_reasons: typing.Set[str] = set()

        # Seconds to wait for a result from transcriber thread
        self.session_result_timeout = session_result_timeout

        def default_recorder():
            return WebRtcVadRecorder(
                max_seconds=max_seconds,
                vad_mode=vad_mode,
                skip_seconds=skip_seconds,
                min_seconds=min_seconds,
                speech_seconds=speech_seconds,
                silence_seconds=silence_seconds,
                before_seconds=before_seconds,
                silence_method=silence_method,
                current_energy_threshold=current_energy_threshold,
                max_energy=max_energy,
                max_current_ratio_threshold=max_current_energy_ratio_threshold,
            )

        self.recorder_factory = recorder_factory or default_recorder

        self.first_audio: bool = True

        # WAV buffers for each session
        self.sessions: typing.Dict[typing.Optional[str], TranscriberInfo] = {}
        self.free_transcribers: typing.List[TranscriberInfo] = []

    # -------------------------------------------------------------------------

    async def start_listening(
        self, message: AsrStartListening
    ) -> typing.AsyncIterable[typing.Union[StopListeningType, AsrError]]:
        """Start recording audio data for a session."""
        try:
            if message.session_id in self.sessions:
                # Stop existing session
                async for stop_message in self.stop_listening(
                    AsrStopListening(
                        site_id=message.site_id, session_id=message.session_id
                    )
                ):
                    yield stop_message

            if self.free_transcribers:
                # Re-use existing transcriber
                info = self.free_transcribers.pop()

                _LOGGER.debug(
                    "Re-using existing transcriber (session_id=%s)", message.session_id
                )
            else:
                # Create new transcriber
                info = TranscriberInfo(reuse=self.reuse_transcribers)
                _LOGGER.debug("Creating new transcriber session %s", message.session_id)

                def transcribe_proc(
                    info, transcriber_factory, sample_rate, sample_width, channels
                ):
                    def audio_stream(frame_queue) -> typing.Iterable[bytes]:
                        # Pull frames from the queue
                        frames = frame_queue.get()
                        while frames:
                            yield frames
                            frames = frame_queue.get()

                    try:
                        info.transcriber = transcriber_factory()

                        assert (
                            info.transcriber is not None
                        ), "Failed to create transcriber"

                        while True:
                            # Wait for session to start
                            info.ready_event.wait()
                            info.ready_event.clear()

                            # Get result of transcription
                            result = info.transcriber.transcribe_stream(
                                audio_stream(info.frame_queue),
                                sample_rate,
                                sample_width,
                                channels,
                            )

                            _LOGGER.debug("Transcription result: %s", result)

                            assert (
                                result is not None and result.text
                            ), "Null transcription"

                            # Signal completion
                            info.result = result
                            info.result_event.set()

                            if not self.reuse_transcribers:
                                try:
                                    info.transcriber.stop()
                                except Exception:
                                    _LOGGER.exception("Transcriber stop")

                                break
                    except Exception:
                        _LOGGER.exception("session proc")

                        # Mark as not reusable
                        info.reuse = False

                        # Stop transcriber
                        if info.transcriber is not None:
                            try:
                                info.transcriber.stop()
                            except Exception:
                                _LOGGER.exception("Transcriber stop")

                        # Signal failure
                        info.transcriber = None
                        info.result = Transcription(
                            text="", likelihood=0, transcribe_seconds=0, wav_seconds=0
                        )
                        info.result_event.set()

                # Run in separate thread
                info.thread = threading.Thread(
                    target=transcribe_proc,
                    args=(
                        info,
                        self.transcriber_factory,
                        self.sample_rate,
                        self.sample_width,
                        self.channels,
                    ),
                    daemon=True,
                )

                info.thread.start()

            # ---------------------------------------------------------------------

            # Settings for session
            info.start_listening = message

            # Signal session thread to start
            info.ready_event.set()

            if message.stop_on_silence:
                # Begin silence detection
                if info.recorder is None:
                    info.recorder = self.recorder_factory()

                info.recorder.start()
            else:
                # Use internal buffer (no silence detection)
                info.audio_buffer = bytes()

            self.sessions[message.session_id] = info
            _LOGGER.debug("Starting listening (session_id=%s)", message.session_id)
            self.first_audio = True
        except Exception as e:
            _LOGGER.exception("start_listening")
            yield AsrError(
                error=str(e),
                context=repr(message),
                site_id=message.site_id,
                session_id=message.session_id,
            )

    async def stop_listening(
        self, message: AsrStopListening
    ) -> typing.AsyncIterable[StopListeningType]:
        """Stop recording audio data for a session."""
        info = self.sessions.pop(message.session_id, None)
        if info:
            try:
                # Trigger publishing of transcription on end of session
                async for result in self.finish_session(
                    info, message.site_id, message.session_id
                ):
                    yield result

                if info.reuse and (info.transcriber is not None):
                    # Reset state
                    info.result = None
                    info.result_event.clear()
                    info.result_sent = False
                    info.result = None
                    info.start_listening = None
                    info.audio_buffer = None

                    while info.frame_queue.qsize() > 0:
                        info.frame_queue.get_nowait()

                    # Add to free pool
                    self.free_transcribers.append(info)
            except Exception as e:
                _LOGGER.exception("stop_listening")
                yield AsrError(
                    error=str(e),
                    context=repr(info.transcriber),
                    site_id=message.site_id,
                    session_id=message.session_id,
                )

        _LOGGER.debug("Stopping listening (session_id=%s)", message.session_id)

    async def handle_audio_frame(
        self,
        frame_wav_bytes: bytes,
        site_id: str = "default",
        session_id: typing.Optional[str] = None,
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrRecordingFinished,
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, TopicArgs],
        ]
    ]:
        """Process single frame of WAV audio"""

        # Don't process audio if no sessions
        if not self.sessions:
            return

        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        if session_id is None:
            # Add to every open session
            target_sessions = list(self.sessions.items())
        else:
            # Add to single session
            target_sessions = [(session_id, self.sessions[session_id])]

        # Add to every open session with matching site_id
        for target_id, info in target_sessions:
            try:
                assert info.start_listening is not None

                # Match site_id
                if info.start_listening.site_id != site_id:
                    continue

                # Push to transcription thread
                info.frame_queue.put(audio_data)

                if info.recorder is not None:
                    # Check for voice command end
                    command = info.recorder.process_chunk(audio_data)

                    if command:
                        # Trigger publishing of transcription on silence
                        async for result in self.finish_session(
                            info, site_id=site_id, session_id=target_id
                        ):
                            yield result
                else:
                    # Use session audio buffer
                    assert info.audio_buffer is not None
                    info.audio_buffer += audio_data
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(info.transcriber),
                    site_id=site_id,
                    session_id=target_id,
                )

    async def finish_session(
        self, info: TranscriberInfo, site_id: str, session_id: typing.Optional[str]
    ) -> typing.AsyncIterable[
        typing.Union[AsrRecordingFinished, AsrTextCaptured, AudioCapturedType]
    ]:
        """Publish transcription result for a session if not already published"""

        if info.recorder is not None:
            # Stop silence detection and get trimmed audio
            audio_data = info.recorder.stop()
        else:
            # Use complete audio buffer
            assert info.audio_buffer is not None
            audio_data = info.audio_buffer

        if not info.result_sent:
            # Send recording finished message
            yield AsrRecordingFinished(site_id=site_id, session_id=session_id)

            # Avoid re-sending transcription
            info.result_sent = True

            # Last chunk
            info.frame_queue.put(None)

            # Wait for result
            result_success = info.result_event.wait(timeout=self.session_result_timeout)
            if not result_success:
                # Mark transcription as non-reusable
                info.reuse = False

            transcription = info.result
            assert info.start_listening is not None

            if transcription:
                # Successful transcription
                yield (
                    AsrTextCaptured(
                        text=transcription.text,
                        likelihood=transcription.likelihood,
                        seconds=transcription.transcribe_seconds,
                        site_id=site_id,
                        session_id=session_id,
                        lang=info.start_listening.lang,
                    )
                )
            else:
                # Empty transcription
                yield AsrTextCaptured(
                    text="",
                    likelihood=0,
                    seconds=0,
                    site_id=site_id,
                    session_id=session_id,
                    lang=info.start_listening.lang,
                )

            if info.start_listening.send_audio_captured:
                wav_bytes = self.to_wav_bytes(audio_data)

                # Send audio data
                yield (
                    # pylint: disable=E1121
                    AsrAudioCaptured(wav_bytes),
                    {"site_id": site_id, "session_id": session_id},
                )

    # -------------------------------------------------------------------------

    async def handle_train(
        self, train: AsrTrain, site_id: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system"""
        try:
            if (
                not self.no_overwrite_train
                and self.language_model_path
                and self.scorer_path
                and self.alphabet_path
            ):
                _LOGGER.debug("Loading %s", train.graph_path)
                with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                    graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

                # Generate language model/scorer
                _LOGGER.debug("Starting training")
                rhasspyasr_deepspeech.train(
                    graph=graph,
                    language_model=self.language_model_path,
                    scorer_path=self.scorer_path,
                    alphabet_path=self.alphabet_path,
                    base_language_model_fst=self.base_language_model_fst,
                    base_language_model_weight=self.base_language_model_weight,
                    mixed_language_model_fst=self.mixed_language_model_fst,
                )
            else:
                _LOGGER.warning("Not overwriting language model/scorer")

            # Clear out existing transcribers so models can reload on next voice command
            self.free_transcribers = []
            for info in self.sessions.values():
                info.reuse = False

            yield (AsrTrainSuccess(id=train.id), {"site_id": site_id})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield AsrError(
                error=str(e),
                context="handle_train",
                site_id=site_id,
                session_id=train.id,
            )

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        site_id: typing.Optional[str] = None,
        session_id: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        # Check enable/disable messages
        if isinstance(message, AsrToggleOn):
            if message.reason == AsrToggleReason.UNKNOWN:
                # Always enable on unknown
                self.disabled_reasons.clear()
            else:
                self.disabled_reasons.discard(message.reason)

            if self.disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.disabled_reasons)
            else:
                self.enabled = True
                self.first_audio = True
                _LOGGER.debug("Enabled")
        elif isinstance(message, AsrToggleOff):
            self.enabled = False
            self.disabled_reasons.add(message.reason)
            _LOGGER.debug("Disabled")
        elif isinstance(message, AudioFrame):
            if self.enabled:
                assert site_id, "Missing site_id"
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                # Add to all active sessions
                async for frame_result in self.handle_audio_frame(
                    message.wav_bytes, site_id=site_id
                ):
                    yield frame_result
        elif isinstance(message, AudioSessionFrame):
            if self.enabled:
                assert site_id and session_id, "Missing site_id or session_id"
                if session_id in self.sessions:
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to specific session only
                    async for session_frame_result in self.handle_audio_frame(
                        message.wav_bytes, site_id=site_id, session_id=session_id
                    ):
                        yield session_frame_result
        elif isinstance(message, AsrStartListening):
            # hermes/asr/startListening
            async for start_result in self.start_listening(message):
                yield start_result
        elif isinstance(message, AsrStopListening):
            # hermes/asr/stopListening
            async for stop_result in self.stop_listening(message):
                yield stop_result
        elif isinstance(message, AsrTrain):
            # rhasspy/asr/<site_id>/train
            assert site_id, "Missing site_id"
            async for train_result in self.handle_train(message, site_id=site_id):
                yield train_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)
