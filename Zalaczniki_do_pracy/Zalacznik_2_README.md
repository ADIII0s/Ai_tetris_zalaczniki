# Agent AI do gry w Tetris

Projekt przedstawia agenta heurystycznego grającego w Tetrisa. Agent nie uczy się podczas pojedynczej rozgrywki. Najpierw wybiera ruch na podstawie funkcji oceny planszy, a następnie algorytm genetyczny dobiera wagi tej funkcji w kolejnych generacjach.

Projekt powstał jako część pracy dyplomowej dotyczącej optymalizacji strategii agenta grającego w Tetrisa za pomocą algorytmu genetycznego.

## Cel projektu

Celem projektu było:

- uruchomienie środowiska Tetris w Pythonie,
- przygotowanie agenta wybierającego najlepsze ustawienie klocka,
- zbudowanie funkcji oceny planszy,
- automatyczne dobranie wag tej funkcji algorytmem genetycznym,
- porównanie kolejnych wersji agenta,
- zapis wyników, historii treningu i checkpointów.

## Wersje agenta

### Wersja bazowa

Pierwszy agent korzystał z czterech ręcznie dobranych wag:

- suma wysokości kolumn,
- maksymalna wysokość,
- liczba dziur,
- nierówność powierzchni.

Był to punkt odniesienia do dalszych porównań.

### V1

Wersja V1 wykorzystywała nadal cztery cechy planszy, ale ich wagi były dobierane algorytmem genetycznym.

Wynik średni wzrósł z około 7,78 do 19,78 usuniętych linii.

### V2

V2 uporządkowała proces treningu. Dodano:

- większą populację,
- walidację,
- checkpoint,
- zapis historii treningu,
- końcowy test na nowych seedach.

V2 osiągnęła średnio 20,41 usuniętych linii w teście końcowym.

### V3

V3 rozszerzyła opis planszy do 11 cech i 12 wag decyzyjnych. Dodano między innymi:

- głębokość dziur,
- bloki znajdujące się nad dziurami,
- studnie,
- przejścia w wierszach,
- przejścia w kolumnach,
- strefę zagrożenia,
- premię za usunięcie linii.

Końcowa wersja V3 osiągnęła średnio 59,864 usuniętych linii w teście końcowym na 250 nowych seedach.

## Struktura plików

```text
AiTetris/
├── main.py
├── tetris_features_v3.py
├── optimize_weights_v3.py
├── optimize_weights_v2.py
├── benchmark.py
├── compare_agents.py
├── analyze_results.py
├── best_weights.json
├── best_weights_v2.json
├── best_weights_v3.json
├── final_report_v2.txt
├── final_report_v3.txt
├── training_history_v2.csv
├── training_history_v3.csv
├── agent_comparison.csv
├── benchmark_results.csv
├── requirements.txt
└── charts/
    ├── comparison_lines.png
    ├── comparison_pieces.png
    ├── results_boxplot.png
    └── paired_differences.png
```

## Opis najważniejszych plików

| Plik | Opis |
|---|---|
| `main.py` | Uruchamia agenta korzystającego z najlepszych zapisanych wag. |
| `tetris_features_v3.py` | Oblicza 11 cech planszy wykorzystywanych w V3. |
| `optimize_weights_v3.py` | Finalny algorytm genetyczny optymalizujący 12 wag. |
| `optimize_weights_v2.py` | Wcześniejszy optymalizator oparty na 4 cechach. |
| `benchmark.py` | Testuje agenta losowego i prostego agenta heurystycznego. |
| `compare_agents.py` | Porównuje wyniki różnych wersji agenta. |
| `analyze_results.py` | Wykonuje analizę statystyczną i generuje wykresy. |
| `best_weights_v3.json` | Najlepszy zestaw wag V3. |
| `training_history_v3.csv` | Historia treningu V3. |
| `final_report_v3.txt` | Raport końcowy V3 z walidacji i testu. |
| `checkpoint_v3.pkl` | Plik umożliwiający wznowienie treningu V3. |

## Wymagania

Projekt był uruchamiany lokalnie w systemie Windows z poziomu PowerShell.

Wymagane biblioteki znajdują się w pliku `requirements.txt`.

Przykładowa zawartość:

```text
gymnasium==1.3.0
tetris-gymnasium==0.2.1
numpy==1.26.4
matplotlib
scipy
```

## Instalacja

W katalogu projektu uruchom:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jeżeli nie używasz pliku `requirements.txt`, biblioteki można zainstalować ręcznie:

```powershell
python -m pip install gymnasium tetris-gymnasium numpy matplotlib scipy
```

## Uruchomienie najlepszego agenta

Aby uruchomić agenta z najlepszymi wagami:

```powershell
python .\main.py
```

Program wczyta plik `best_weights_v3.json` lub inny plik wag wskazany w kodzie i rozpocznie rozgrywkę.

## Uruchomienie treningu V3

Aby rozpocząć lub wznowić trening V3:

```powershell
python .\optimize_weights_v3.py
```

Jeżeli w katalogu znajduje się plik:

```text
checkpoint_v3.pkl
```

program wznowi trening od zapisanego punktu.

Jeżeli checkpoint nie istnieje, program utworzy nową populację i zacznie trening od początku.

## Mechanizm checkpointu

Checkpoint zapisuje:

- populację osobników,
- numer następnej generacji,
- najlepszy znaleziony zestaw wag,
- historię treningu,
- stan generatora liczb losowych.

Dzięki temu można przerwać trening i później kontynuować go bez rozpoczynania od zera.

## Wyniki końcowe V3

Końcowy trening V3 zakończył się następującymi wynikami:

| Metryka | Walidacja | Test końcowy |
|---|---:|---:|
| Fitness | 72,6510 | 70,3163 |
| Średnia liczba linii | 61,4500 | 59,8640 |
| Mediana liczby linii | 58,0000 | 55,0000 |
| Odchylenie standardowe | 22,4324 | 24,5894 |
| Średnia liczba klocków | 189,3833 | 185,5360 |
| Najlepszy wynik | 134 | 147 |
| Najgorszy wynik | 28 | 17 |
| Udział wyników zerowych | 0,0000 | 0,0000 |

## Najlepsze wagi V3

```text
aggregate_height      0.0470951233
max_height            0.0019709448
holes                 0.0894708173
bumpiness             0.0049065980
hole_depth            0.0249737688
blocks_above_holes    0.0072736397
wells                 0.5564624836
row_transitions       0.0646732369
column_transitions    0.0700320185
danger_zone           0.0457574020
filled_cells          0.0518301394
line_clear_reward     0.0355538278
```

Największą wagę otrzymała cecha `wells`, czyli miara studni i wąskich zagłębień. Oznacza to, że algorytm uznał ich kontrolowanie za bardzo ważne dla utrzymania gry.

## Porównanie wersji

| Wersja | Opis | Średnia liczba linii |
|---|---|---:|
| Bazowa | 4 cechy, wagi ręczne | 7,78 |
| V1 | 4 cechy, wagi optymalizowane | 19,78 |
| V2 | 4 cechy, pełniejsza walidacja | 20,41 |
| V3 | 11 cech i 12 wag | 59,864 |

Największa poprawa pojawiła się pomiędzy V2 i V3. Oznacza to, że samo dalsze strojenie czterech wag miało ograniczony efekt. Dużo większe znaczenie miało rozszerzenie opisu planszy.

## Uwagi techniczne

- Projekt nie wykorzystuje obrazu z ekranu.
- Agent nie steruje klawiszami krok po kroku.
- Środowisko generuje legalne ustawienia klocka, a agent wybiera najlepsze z nich.
- Agent nie planuje kilku klocków do przodu.
- Algorytm genetyczny nie tworzy nowych cech, tylko dobiera wagi cech przygotowanych w kodzie.
- Wyniki zależą od liczby generacji, liczby seedów oraz mocy komputera.

## Autor

Adrian Rachubiński  
Kierunek: Informatyka  
Projekt dyplomowy, Poznań 2026
