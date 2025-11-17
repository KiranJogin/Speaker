# 🎙️ SpeakerSplit – AI Voice Separation System
## 🔊 AI-Powered Multi-Speaker Voice Separation & Playback

SpeakerSplit is an AI-based audio processing application designed to separate mixed audio recordings containing multiple speakers into clean, isolated audio tracks.
Using advanced signal processing and machine learning clustering techniques, the system identifies different speakers, segments the audio, and reconstructs individual speaker tracks that users can play or download.

Built with **Streamlit** for the interface and leveraging powerful audio processing libraries like **librosa**, **scipy**, and **scikit-learn**, SpeakerSplit enables seamless audio upload, configuration, playback, and visual analysis.

---

## 🚀 Features
- Separate multiple overlapping voices from a single audio source
- Automatic speaker identification & clustering (2–6 speakers supported)
- Noise reduction, audio normalization & spectral filtering
- Individual speaker track export & playback dashboard
- Visual analysis: waveform & spectrogram preview
- User-configurable settings (noise level, speaker count, chunk size)

---

## 🏗 System Architecture

### **Frontend (UI Layer) – Streamlit Application**
| Component | Description |
|-----------|------------|
| Framework | Streamlit |
| Layout | Multi-tab interface |
| Tabs | Upload & Process, Playback Dashboard, Analysis |
| State Management | Session state for audio, separated tracks, sample rate |
| Configurable Controls | Sidebar for noise reduction, normalization, speaker count |

---

### **Backend Architecture**
| Folder / Module | Responsibility |
|-----------------|----------------|
| `audio_processor.py` | Preprocessing (noise removal, normalization, filtering) |
| `speaker_separator.py` | Feature extraction, clustering, separation logic |
| `app.py` | Main orchestrator & UI integration |

### **Processing Pipeline**
1. Upload audio & convert format (`soundfile` / `pydub`)
2. Apply preprocessing (noise profile estimation, normalization)
3. Extract audio features (MFCC, spectral contrast, chroma, ZCR)
4. Segment audio into time-based chunks (~1 second)
5. Cluster speaker segments (`KMeans` / hierarchical clustering)
6. Rebuild separated tracks for each speaker
7. Output playback & download options

---

## 🔬 Core Signal Processing Methods
| Technique | Purpose |
|-----------|---------|
| STFT (Short-Time Fourier Transform) | Time–frequency analysis |
| Median-based noise profiling | Background noise reduction |
| Band-pass filtering (80 Hz – 8 kHz) | Human voice isolation |
| Spectral masking | Remove noise outside threshold |
| 1-second chunk segmentation | Granular clustering per speaker |

---

## 🧠 Machine Learning Components
| Component | Description |
|-----------|-------------|
| Feature Extraction | MFCC (13), spectral contrast (7), chroma (12), ZCR |
| Clustering Algorithms | KMeans & Hierarchical clustering |
| Distance Metrics | Cosine / Euclidean |
| Speaker Count | Configurable (2–6 speakers) |

---

## 📂 Data Storage Strategy
- Temporary storage via `tempfile`
- In-memory processing via NumPy arrays
- Streamlit session state persistence
- No database or permanent storage

---

## 📦 Dependencies

### **Audio & Signal Processing**
- `librosa`
- `soundfile`
- `pydub`
- `scipy.signal`
- `numpy`

### **Machine Learning & Clustering**
- `scikit-learn`
- `scipy.cluster.hierarchy`
- `scipy.spatial.distance`

### **UI & Visualization**
- `streamlit`
- `matplotlib`

---

## 🖥 User Interface Capabilities
- Streamlit-based interactive web dashboard
- Upload audio, view spectrogram & waveforms
- Per-speaker playback and download controls
- Sidebar settings for intelligent configuration

---

## ⚙️ Audio Parameters
| Parameter | Default |
|-----------|---------|
| Sample Rate | Auto-managed by `librosa` |
| FFT Size | 2048 |
| Hop Length | 512 |
| Noise Threshold | 1.5× baseline |
| Normalization | 95% peak limit |

---

## 📌 Use Cases
- Podcast multi-speaker audio editing
- Meeting / Lecture separation for clarity
- Interview audio enhancement
- Audio forensics & evidence analysis
- Post-production for content creators

---

## 🏁 Future Enhancements
- Name-based speaker labeling
- Real-time microphone streaming
- Speech-to-text & summaries per speaker
- Automated transcription export

