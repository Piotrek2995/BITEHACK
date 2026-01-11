# ❄️ Skrzyczne Snow Coverage AI

Aplikacja do monitorowania pokrywy śnieżnej na stokach narciarskich Skrzycznego z wykorzystaniem danych satelitarnych Sentinel-2 i wizualizacji 3D w CesiumJS.

## 🎯 Funkcje

- **📡 Analiza satelitarna** – pobieranie i przetwarzanie danych Sentinel-2 z Microsoft Planetary Computer
- **🗺️ Mapy dzienne** – pokrycie śniegiem dla każdego dnia sezonu
- **📊 Mapa sezonowa** – prawdopodobieństwo występowania śniegu (agregacja wielu dni)
- **⛷️ Stoki narciarskie** – wizualizacja tras z podziałem na trudność
- **🌍 Wizualizacja 3D** – interaktywna mapa w CesiumJS z terenem

## 🛠️ Technologie

| Komponent | Technologia |
|-----------|-------------|
| Dane satelitarne | Sentinel-2 L2A, Microsoft Planetary Computer |
| Przetwarzanie | Python, stackstac, rioxarray, NumPy |
| Indeks śniegu | NDSI (Normalized Difference Snow Index) |
| Wizualizacja | CesiumJS, GeoJSON |
| Format danych | GeoTIFF, PNG, JSON |

## 📁 Struktura projektu

```
BITEHACK/
├── main.py              # Główny skrypt analizy śniegu
├── ndsi_daily.py        # Pobieranie dziennych map NDSI
├── ndsi_to_png.py       # Konwersja TIFF → PNG + mapa sezonowa
├── cesium/
│   ├── index.html       # Interfejs webowy
│   ├── stats.json       # Statystyki pokrycia
│   ├── png_daily2/      # Mapy dzienne (PNG)
│   └── trasy_skrzyczne.geojson
├── ndsi_daily2/         # Surowe dane NDSI (TIFF)
└── skrzyczne/           # Shapefile obszaru
```

## 🚀 Uruchomienie

### 1. Instalacja zależności

```bash
pip install planetary-computer pystac-client stackstac rioxarray numpy pillow rasterio
```

### 2. Pobieranie danych satelitarnych

```bash
python ndsi_daily.py
```

### 3. Generowanie map PNG

```bash
python ndsi_to_png.py
```

### 4. Uruchomienie wizualizacji

```bash
cd cesium
python -m http.server 8000
```

Otwórz w przeglądarce: http://localhost:8000

## 📐 Parametry

| Parametr | Wartość | Opis |
|----------|---------|------|
| BBOX | `[18.97, 49.67, 19.06, 49.71]` | Obszar Skrzycznego |
| NDSI_THRESHOLD | `0.4` | Próg klasyfikacji śniegu |
| MAX_CLOUD | `30%` | Maksymalne zachmurzenie |
| RESOLUTION | `20m` | Rozdzielczość przestrzenna |

## 🎨 Legenda map

### Mapy dzienne
- ⚪ **Biały** – śnieg (NDSI ≥ 0.4)
- 🔲 **Przezroczysty** – brak śniegu

### Mapa sezonowa
- ⚪ **Biały** – pewny śnieg (70%+ dni ze śniegiem)
- ⚫ **Szary** – częsty śnieg (30-70% dni)

### Stoki narciarskie
- 🟢 **Zielony** – łatwe (easy/novice)
- 🔵 **Niebieski** – średnie (intermediate)
- 🔴 **Czerwony** – trudne (advanced)
- ⚫ **Czarny** – eksperckie (expert)

## 📡 Źródła danych

- **Sentinel-2 L2A** – ESA/Copernicus via Microsoft Planetary Computer
- **Teren 3D** – Cesium World Terrain
- **Trasy narciarskie** – OpenStreetMap

## 👥 Autorzy
Piotr Pawlus, Szymon Ziedalski, Bartosz Ziolkowski, Mateusz Stelmasiak
Projekt stworzony na hackathonie **BITEHACK 2026**.

## 📄 Licencja

MIT License
