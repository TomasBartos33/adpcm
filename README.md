# ADPCM Speech Coder - Python/PyQt6

Semestralne vypracovanie prevadza Matlab GUI aplikaciu `adpcm_GUI25.m` do Pythonu.

## Spustenie

### macOS / Linux

```bash
./run_app.sh
```

Skript pri prvom spusteni vytvori lokalne virtualne prostredie `venv`, nainstaluje zavislosti z `requirements.txt` a spusti aplikaciu.

Rucny postup pre macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m adpcm_py.main
```

Ak uz su baliky vo `venv` nainstalovane, staci:

```bash
source venv/bin/activate
python -m adpcm_py.main
```

### Windows

Najjednoduchsie spustenie:

```bat
run_app.bat
```

Skript pri prvom spusteni vytvori lokalne virtualne prostredie `venv`, nainstaluje zavislosti z `requirements.txt` a spusti aplikaciu.

Rucny postup pre Windows:

```bat
py -3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m adpcm_py.main
```

Ak prikaz `py` nie je dostupny, pouzite:

```bat
python -m venv venv
```

## Funkcionalita

- nacitanie WAV suboru,
- nastavenie parametrov ADPCM kodera (`nbits`, `alpha`, `deltamin`, `deltamax`),
- adaptivna kvantizacia podla Jayantovej metody,
- vypocet SNR,
- graf signal/error spektra, histogram chyby a priebehy `x`, `xhat`, `error`,
- prehratie povodneho, kodovaneho a chyboveho signalu,
- ulozenie vystupov do `adpcm_py/output`.

V povodnom ZIP-e nebol samostatny vstupny speech WAV z priecinka `speech_files`, preto je v `adpcm_py/data/hcdr05.wav` pripraveny ukazkovy WAV na okamzite vyskusanie aplikacie.

## Vyvojove prostredie

Vyvoj a testovanie prebiehali na MacBook Pro M4 (Apple Silicon, macOS 26 Tahoe), 48 GB RAM, 12-jadrovym CPU a 16-jadrovym GPU.
