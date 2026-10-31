"""The score of the film, synthesized: python -I tools/music.py out/music.wav

3:29 in D minor at 60 BPM (one beat per second, so cue times are seconds), following the shots:
  0-21    cloud descent       wind, a low D drone swelling out of silence
  18-43   establishing view   string pad enters (Dm add9), distant bells
  40-63   street level        a slow sub pulse like footsteps, rain hiss
  60-103  plaza, crane        the progression Dm - Bb - F - Csus rises, a bell melody
  100-146 cribs, flight       soft arpeggio, the progression continues, more air
  143-169 sun break           lift to Bb major / F, warm swell, the climax at ~156 s
  166-190 dusk                back to D minor, sparse low piano
  187-209 night, title        drone, glass notes, a last chord dying away by 208 s
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000
DUR = 209.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
L = np.zeros(N)
R = np.zeros(N)
REV = np.zeros((2, N))   # send to the reverb


def hz(note):
    names = {"C": 0, "C#": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
    n, o = note[:-1], int(note[-1])
    return 440.0 * 2 ** ((names[n] + 12 * (o + 1) - 69) / 12)


def env(n, a, r, sus=1.0):
    """Attack / release envelope (seconds) over n samples, raised-cosine edges."""
    e = np.full(n, sus)
    na, nr = min(int(a * SR), n), min(int(r * SR), n)
    if na > 0:
        e[:na] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, na))
    if nr > 0:
        e[n - nr:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nr))
    return e


def put(sig, start, pan=0.0, gain=1.0, rev=0.4):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i:i + len(sig)] += sig * gl * (1 - rev * 0.5)
    R[i:i + len(sig)] += sig * gr * (1 - rev * 0.5)
    REV[0, i:i + len(sig)] += sig * gl * rev
    REV[1, i:i + len(sig)] += sig * gr * rev


def pad(freqs, start, dur, a=4.0, r=5.0, bright=900.0, gain=0.05, pan=0.0):
    """Detuned, softly filtered saw ensemble (additive), slow bow-like swell."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    sweep = 1 + 0.35 * np.sin(2 * np.pi * t / max(dur, 1) * 0.9)
    for f in freqs:
        for det in (-0.11, 0.0, 0.13):
            ph = rng.uniform(0, 2 * np.pi)
            vib = 0.0025 * np.sin(2 * np.pi * (4.6 + det) * t + ph)
            f0 = f * 2 ** (det / 12)
            k = 1
            while f0 * k < 6000 and k < 40:
                w = (1.0 / k) * np.exp(-f0 * k / (bright * sweep))
                out += w * np.sin(2 * np.pi * f0 * k * t * (1 + vib) + ph * k)
                k += 1
    out *= env(n, a, r) / (len(freqs) * 3)
    put(out, start, pan, gain, rev=0.55)


def bell(f, start, gain=0.05, pan=0.0, decay=6.0):
    n = int((decay + 1) * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for ratio, amp, d in ((1.0, 1.0, 1.0), (2.0, 0.5, 0.7), (2.76, 0.35, 0.5), (5.4, 0.18, 0.3), (8.93, 0.08, 0.2)):
        out += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / (decay * d))
    out *= env(n, 0.004, 0.5)
    put(out, start, pan, gain, rev=0.7)


def piano(f, start, gain=0.06, pan=0.0, decay=4.0):
    """A soft felt-piano-like tone: a few slightly stretched partials, fast hammer, long fade."""
    n = int((decay + 1.5) * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, amp in ((1, 1.0), (2, 0.42), (3, 0.16), (4, 0.08), (5, 0.04)):
        fk = f * k * (1 + 0.0004 * k * k)
        out += amp * np.sin(2 * np.pi * fk * t) * np.exp(-t * (0.6 + 0.5 * k) / decay * 2)
    out *= env(n, 0.008, 1.0)
    put(out, start, pan, gain, rev=0.6)


def pluck(f, start, gain=0.03, pan=0.0):
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    out = np.sin(2 * np.pi * f * t) * np.exp(-t * 3.5) + 0.3 * np.sin(4 * np.pi * f * t) * np.exp(-t * 6)
    out *= env(n, 0.003, 0.3)
    put(out, start, pan, gain, rev=0.65)


def thump(start, gain=0.25):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    f = 52 + 30 * np.exp(-t * 25)
    out = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6)
    out *= env(n, 0.002, 0.2)
    put(out, start, 0.0, gain, rev=0.2)


def drone(f, start, dur, gain=0.08, a=6.0, r=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, amp in ((1, 1.0), (2, 0.35), (3, 0.12), (4, 0.06)):
        out += amp * np.sin(2 * np.pi * f * k * t + 0.4 * np.sin(2 * np.pi * 0.07 * k * t))
    out *= (0.8 + 0.2 * np.sin(2 * np.pi * 0.05 * t)) * env(n, a, r)
    put(out, start, 0.0, gain, rev=0.3)


def noise_band(start, dur, lo, hi, gain, a=4.0, r=4.0, swell=0.08):
    n = int(dur * SR)
    sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
    t = np.arange(n) / SR
    for ch, pan in ((0, -0.6), (1, 0.6)):
        x = sosfilt(sos, rng.standard_normal(n))
        x *= (0.6 + 0.4 * np.sin(2 * np.pi * swell * t + ch * 1.7)) * env(n, a, r)
        put(x / (np.std(x) + 1e-9), start, pan, gain, rev=0.3)


# ---------------------------------------------------------------- the cues
# wind through the whole film, louder in the clouds and at the end
noise_band(0.0, 209.0, 180, 900, 0.0045, a=5.0, r=6.0)
noise_band(0.0, 30.0, 300, 2400, 0.0065, a=6.0, r=10.0, swell=0.13)
noise_band(185.0, 24.0, 200, 1500, 0.005, a=6.0, r=8.0)

# the low D under everything but the sun break
drone(hz("D2"), 0.5, 146.0, gain=0.09, a=10.0, r=8.0)
drone(hz("A1"), 30.0, 115.0, gain=0.04, a=10.0, r=10.0)
drone(hz("D2"), 165.0, 43.5, gain=0.09, a=6.0, r=9.0)

# 1-2: a first pad from the cloud, then Dm add9 for the establishing view; far bells
pad([hz("D3"), hz("A3")], 6.0, 18.0, a=8.0, r=6.0, bright=500, gain=0.10)
pad([hz("D3"), hz("F3"), hz("A3"), hz("E4")], 19.0, 25.0, a=5.0, r=6.0, bright=700, gain=0.12)
for t0, nt, pn in ((24.0, "A4", -0.5), (31.0, "E5", 0.4), (37.0, "D5", -0.2)):
    bell(hz(nt), t0, gain=0.035, pan=pn)

# 3: street level: footsteps-slow sub pulse and hiss of the wet road
for k in range(21):
    thump(41.0 + k * 1.0, gain=0.10 + 0.10 * min(k / 6, 1) * (1 - max(0, k - 16) / 5))
noise_band(40.0, 23.0, 2500, 9000, 0.002, a=3.0, r=4.0, swell=0.2)
pad([hz("D3"), hz("A3"), hz("C4")], 42.0, 21.0, a=4.0, r=5.0, bright=600, gain=0.09)

# 4-6: the progression rises: Dm - Bb - F - Csus, 6 s each, twice; bell melody over it
PROG = [["D3", "F3", "A3", "D4"], ["Bb2", "F3", "Bb3", "D4"], ["F2", "C3", "A3", "C4"], ["C3", "G3", "C4", "F4"]]
t = 60.0
for rep in range(4):
    for ch in PROG:
        g = 0.09 + 0.02 * rep
        pad([hz(n) for n in ch], t, 7.5, a=2.5, r=3.5, bright=650 + 120 * rep, gain=g, pan=-0.15 + 0.1 * rep)
        t += 6.0
    if t > 140:
        break
MEL = [(62, "A4"), (65, "F4"), (68, "D5"), (74, "C5"), (77, "Bb4"), (80, "A4"), (86, "A4"), (89, "C5"), (92, "D5"),
       (98, "E5"), (101, "F5"), (104, "E5"), (110, "D5"), (113, "C5"), (116, "A4"), (122, "A4"), (125, "Bb4"), (128, "C5"),
       (134, "D5"), (137, "C5"), (140, "A4")]
for t0, nt in MEL:
    bell(hz(nt), t0, gain=0.03, pan=0.3 * np.sin(t0))
# arpeggio for the cribs and the flight (100-146): eighths of the chord tones, low and soft
for i in range(int((146 - 100) * 2)):
    t0 = 100.0 + i * 0.5
    ch = PROG[int((t0 - 60) // 6) % 4]
    nt = ch[[0, 2, 1, 3, 2, 1][i % 6]]
    f = hz(nt) * 2
    ramp = min((t0 - 100) / 10, 1) * min((146 - t0) / 4, 1)
    pluck(f, t0, gain=0.022 * ramp, pan=0.5 * np.sin(i * 0.9))

# 8: the sun break: Bb major, F, Gm, F/A lifting; warm swell to ~156 s
BREAK = [(143.0, ["Bb2", "F3", "D4", "A4"]), (149.0, ["F2", "C4", "F4", "A4"]), (155.0, ["G2", "D4", "G4", "Bb4"]),
         (161.0, ["A2", "E4", "A4", "C5"])]
for t0, ch in BREAK:
    pad([hz(n) for n in ch], t0, 8.0, a=3.0, r=4.0, bright=1300, gain=0.15)
drone(hz("Bb1"), 143.0, 14.0, gain=0.06, a=4.0, r=6.0)
for t0, nt in ((147.0, "D5"), (150.0, "F5"), (153.0, "A5"), (156.0, "G5"), (159.0, "F5"), (163.0, "E5")):
    bell(hz(nt), t0, gain=0.035, pan=0.2)

# 9: dusk: D minor, sparse low piano
pad([hz("D3"), hz("F3"), hz("A3")], 166.0, 24.0, a=5.0, r=7.0, bright=450, gain=0.10)
for t0, nt in ((168.0, "D3"), (171.0, "A3"), (174.0, "F3"), (178.0, "E3"), (181.0, "D3"), (184.0, "C3"), (186.5, "A2")):
    piano(hz(nt), t0, gain=0.09, pan=-0.2)
for t0, nt in ((172.5, "A4"), (179.5, "F4"), (185.5, "E4")):
    piano(hz(nt), t0, gain=0.05, pan=0.3)

# 10: night and the title: glass notes, a last open fifth dying away
pad([hz("D3"), hz("A3"), hz("E4")], 187.0, 21.0, a=5.0, r=11.0, bright=500, gain=0.10)
for t0, nt in ((190.0, "A5"), (194.0, "E5"), (197.5, "D5"), (201.0, "A4")):
    bell(hz(nt), t0, gain=0.03, pan=-0.3 + 0.2 * (t0 - 190) / 10, decay=8.0)

# ---------------------------------------------------------------- reverb and master
ir_n = int(5.5 * SR)
ti = np.arange(ir_n) / SR
irs = []
for ch in range(2):
    x = rng.standard_normal(ir_n) * np.exp(-ti / 1.3)
    x = sosfilt(butter(1, 3500, btype="low", fs=SR, output="sos"), x)
    x[: int(0.02 * SR)] = 0
    irs.append(x / np.sqrt(np.sum(x ** 2)))
wetL = fftconvolve(REV[0], irs[0])[:N]
wetR = fftconvolve(REV[1], irs[1])[:N]
mix = np.stack([L + 0.8 * wetL, R + 0.8 * wetR])
mix = sosfilt(butter(2, 28, btype="high", fs=SR, output="sos"), mix)
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
# fade the very end so it is silent by 208.5 s
fade = np.ones(N)
f0, f1 = int(205.0 * SR), int(208.5 * SR)
fade[f0:f1] = np.linspace(1, 0, f1 - f0) ** 2
fade[f1:] = 0
mix *= fade * 10 ** (-1.0 / 20)
pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")

import wave
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", sys.argv[1], f"{N / SR:.3f} s")
