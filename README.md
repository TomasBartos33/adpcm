# ADPCM Speech Coder - Python/PyQt6

Semestralne vypracovanie prevadza Matlab GUI aplikaciu `adpcm_GUI25.m` do Pythonu.

## Spustenie

```bash
source venv/bin/activate
pip install -r requirements.txt
python -m adpcm_py.main
```

Ak uz su baliky vo `venv` nainstalovane, staci:

```bash
source venv/bin/activate
python -m adpcm_py.main
```

Alternativne:

```bash
./run_app.sh
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
