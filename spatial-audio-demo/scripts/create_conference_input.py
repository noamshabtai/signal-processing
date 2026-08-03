"""Render the conference call script into the multi-channel demo input.

Each participant is synthesized separately and written to its own channel, as
if every leg of the call had been recorded locally. Channels therefore contain
no crosstalk: a channel is silent unless its own speaker is talking.
"""

import pathlib
import shutil
import urllib.parse
import urllib.request
import wave

import numpy as np
import piper
import piper.download_voices
import scipy.signal
import yaml

SCRIPT_DIR = pathlib.Path(__file__).parent
SCRIPT_PATH = SCRIPT_DIR / "conference_call.yaml"
VOICES_DIR = SCRIPT_DIR / "voices"
OUTPUT_PATH = SCRIPT_DIR / "../conference_input.wav"
DTYPE = np.int16
PEAK = 0.7
DOWNLOAD_ATTEMPTS = 3


def voice_url(voice, extension):
    """Build the upstream URL for one of a voice's files.

    The scheme is checked rather than trusted: the template comes from piper, so
    an upstream change to it decides what this script would otherwise open, and
    urlopen honours `file:` as readily as `https:`.
    """
    lang_code, voice_name, voice_quality = voice.split("-")
    url = piper.download_voices.URL_FORMAT.format(
        lang_family=lang_code.split("_")[0],
        lang_code=lang_code,
        voice_name=voice_name,
        voice_quality=voice_quality,
        extension=extension,
    )
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError(f"Refusing to fetch a voice over a non-HTTPS URL: {url}")
    return url


def download(voice, extension):
    """Fetch a voice file, verifying the size so a cut transfer is retried.

    Piper's own downloader neither checks the length nor resumes, and silently
    leaves a truncated model behind that only fails later at load time.
    """
    path = VOICES_DIR / f"{voice}{extension}"
    for attempt in range(DOWNLOAD_ATTEMPTS):
        with urllib.request.urlopen(voice_url(voice, extension)) as response:  # noqa: S310  # scheme checked above
            expected_size = int(response.headers["Content-Length"])
            if path.exists() and path.stat().st_size == expected_size:
                return path
            print(f"Downloading {path.name} ({expected_size / 1e6:.0f} MB, attempt {attempt + 1})")
            with open(path, "wb") as voice_file:
                shutil.copyfileobj(response, voice_file)
        if path.stat().st_size == expected_size:
            return path
    raise RuntimeError(f"Could not download {path.name} in one piece after {DOWNLOAD_ATTEMPTS} attempts")


def load_voices(speakers):
    """Download each speaker's voice model if needed and load it."""
    VOICES_DIR.mkdir(exist_ok=True)
    voices = {}
    for name, speaker in speakers.items():
        model_path = download(speaker["voice"], ".onnx")
        download(speaker["voice"], ".onnx.json")
        voices[name] = piper.PiperVoice.load(model_path)
        print(f"Loaded {speaker['voice']} for {name}")
    return voices


def synthesize(voice, text, sampling_frequency):
    """Synthesize text and resample it to the target sampling frequency."""
    chunks = list(voice.synthesize(text))
    audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
    return scipy.signal.resample_poly(audio, sampling_frequency, chunks[0].sample_rate)


def truncate(audio, keep, fade_out, sampling_frequency):
    """Cut an utterance short and fade it, modelling a speaker being cut off."""
    kept_samples = round(len(audio) * keep)
    audio = audio[:kept_samples].copy()
    fade_samples = min(round(fade_out * sampling_frequency), kept_samples)
    audio[kept_samples - fade_samples :] *= np.linspace(1.0, 0.0, fade_samples)
    return audio


def render_utterances(script, voices):
    """Render every utterance, keyed by id."""
    sampling_frequency = script["sampling_frequency"]
    audio_of = {}
    for utterance in script["utterances"]:
        audio = synthesize(voices[utterance["speaker"]], utterance["text"], sampling_frequency)
        if "keep" in utterance:
            audio = truncate(audio, utterance["keep"], script["fade_out"], sampling_frequency)
        audio_of[utterance["id"]] = audio
        print(f"{utterance['id']}: {len(audio) / sampling_frequency:.1f} s")
    return audio_of


def resolve_onsets(script, audio_of):
    """Turn the script's relative timing into absolute onsets, in seconds.

    Utterances anchored with `after` start relative to the *start* of the
    utterance they name and do not advance the chain, so a back-channel can
    land mid-sentence without shifting the turn that follows it.
    """
    sampling_frequency = script["sampling_frequency"]
    onset_of = {}
    chain_end = 0.0
    for utterance in script["utterances"]:
        duration = len(audio_of[utterance["id"]]) / sampling_frequency
        if "after" in utterance:
            onset_of[utterance["id"]] = onset_of[utterance["after"]] + utterance["offset"]
        else:
            onset_of[utterance["id"]] = max(0.0, chain_end + utterance.get("gap", 0.0))
            chain_end = onset_of[utterance["id"]] + duration
    return onset_of


def mix(script, audio_of, onset_of):
    """Lay every utterance onto its speaker's channel at its onset."""
    sampling_frequency = script["sampling_frequency"]
    channel_of = {name: speaker["channel"] for name, speaker in script["speakers"].items()}
    ends = [
        round(onset_of[utterance["id"]] * sampling_frequency) + len(audio_of[utterance["id"]])
        for utterance in script["utterances"]
    ]
    mixed = np.zeros((max(ends), len(channel_of)))
    for utterance in script["utterances"]:
        audio = audio_of[utterance["id"]]
        start = round(onset_of[utterance["id"]] * sampling_frequency)
        mixed[start : start + len(audio), channel_of[utterance["speaker"]]] += audio
    return mixed


def normalize(mixed):
    """Level the channels, as a conference bridge's gain control would."""
    return PEAK * mixed / np.abs(mixed).max(axis=0)


def write(mixed, sampling_frequency):
    """Write the mix as an interleaved multi-channel WAV file."""
    interleaved = (mixed * np.iinfo(DTYPE).max).astype(DTYPE).reshape(-1)
    with wave.open(str(OUTPUT_PATH), "wb") as wav_file:
        wav_file.setnchannels(mixed.shape[1])
        wav_file.setsampwidth(DTYPE().itemsize)
        wav_file.setframerate(sampling_frequency)
        wav_file.writeframes(interleaved.tobytes())


def main():
    """Synthesize the conference call script into a multi-channel WAV file."""
    with open(SCRIPT_PATH, "r") as script_file:
        script = yaml.safe_load(script_file)

    voices = load_voices(script["speakers"])
    audio_of = render_utterances(script, voices)
    onset_of = resolve_onsets(script, audio_of)
    mixed = normalize(mix(script, audio_of, onset_of))
    write(mixed, script["sampling_frequency"])

    duration = len(mixed) / script["sampling_frequency"]
    print(f"Successfully created {OUTPUT_PATH} ({duration:.1f} s, {mixed.shape[1]} channels)")


if __name__ == "__main__":
    main()
