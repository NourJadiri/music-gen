import numpy as np


def make_chord_audio(sr, duration=1.0):
    """Return a mono float32 numpy array: a C major chord with harmonics, vibrato and decay."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)

    # C major chord (C4, E4, G4), each note built from a fundamental + harmonics
    for freq in [261.63, 329.63, 392.00]:
        # phase modulation gives a gentle 5 Hz vibrato (pitch wobble)
        vibrato = 0.5 * np.sin(2 * np.pi * 5 * t)
        for harmonic, amp in [(1, 1.0), (2, 0.5), (3, 0.25)]:
            audio += amp * np.sin(2 * np.pi * freq * harmonic * t + vibrato)

    # volume envelope: quick 50 ms fade-in, then a smooth decay (like a plucked note)
    audio *= np.minimum(t / 0.05, 1.0) * np.exp(-3 * t)

    # a little noise so it isn't perfectly clean
    audio += 0.02 * np.random.randn(len(t))

    # normalize to avoid clipping
    return (0.8 * audio / np.max(np.abs(audio))).astype(np.float32)
