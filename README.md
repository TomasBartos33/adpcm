# ADPCM Speech Coder - Python/PyQt6

Semestrálne vypracovanie prevádza Matlab GUI aplikáciu `adpcm_GUI25.m` do Pythonu.

## Spustenie

### macOS / Linux

```bash
./run_app.sh
```

Skript pri prvom spustení vytvorí lokálne virtuálne prostredie `venv`, nainštaluje závislosti z `requirements.txt` a spustí aplikáciu.

Ručný postup pre macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m adpcm_py.main
```

Ak už sú balíky vo `venv` nainštalované, stačí:

```bash
source venv/bin/activate
python -m adpcm_py.main
```

### Windows

Najjednoduchšie spustenie:

```bat
run_app.bat
```

Skript pri prvom spustení vytvorí lokálne virtuálne prostredie `venv`, nainštaluje závislosti z `requirements.txt` a spustí aplikáciu.

Ručný postup pre Windows:

```bat
py -3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m adpcm_py.main
```

Ak príkaz `py` nie je dostupný, použite:

```bat
python -m venv venv
```

## Funkcionalita

- načítanie WAV súboru,
- nastavenie parametrov ADPCM kodera (`nbits`, `alpha`, `deltamin`, `deltamax`),
- adaptívna kvantizácia podľa Jayantovej metódy,
- výpočet SNR,
- graf signál/error spektra, histogram chyby a priebehy `x`, `xhat`, `error`,
- prehratie pôvodného, kódovaného a chybového signálu,
- uloženie výstupov do `adpcm_py/output`.

V pôvodnom ZIP-e nebol samostatný vstupný speech WAV z priečinka `speech_files`, preto je v `adpcm_py/data/hcdr05.wav` pripravený ukážkový WAV na okamžité vyskúšanie aplikácie.

## Vývojové prostredie

Vývoj a testovanie prebiehali na MacBook Pro M4 (Apple Silicon, macOS 26 Tahoe), 48 GB RAM, 12-jadrovým CPU a 16-jadrovým GPU.
Vývoj a testovanie taktiež prebiehali na PC s 24 GB RAM, 4-jadrovým CPU, 8 GB GPU a operačným systémom Windows 11.
