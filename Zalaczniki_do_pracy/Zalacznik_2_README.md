# Agent AI do gry w Tetris

Projekt przedstawia agenta heurystycznego grającego w Tetrisa. Agent wybiera legalne ustawienie aktualnego klocka na podstawie funkcji oceny planszy, a algorytm genetyczny dobiera wagi tej funkcji w kolejnych generacjach.

Projekt powstał jako część pracy dyplomowej „Optymalizacja strategii agenta grającego w Tetrisa za pomocą algorytmu genetycznego”.

## Cel projektu

Celem projektu było:

- uruchomienie środowiska Tetris w Pythonie,
- przygotowanie agenta oceniającego legalne ustawienia klocka,
- zbudowanie funkcji oceny planszy,
- automatyczne dobranie wag algorytmem genetycznym,
- porównanie wersji bazowej, V1, V2 i V3,
- zapis wyników, historii treningu i checkpointów.

## Wersje agenta

### Wersja bazowa

Agent korzystał z czterech ręcznie dobranych wag: sumy wysokości kolumn, maksymalnej wysokości, liczby dziur i nierówności powierzchni.

### V1

V1 nadal wykorzystywała cztery cechy, ale ich wagi zostały dobrane algorytmem genetycznym. Średni wynik wzrósł z 7,78 do 19,78 usuniętych linii.

### V2

V2 uporządkowała trening przez dodanie walidacji, checkpointu, historii treningu i końcowego testu na nowych seedach. Osiągnęła średnio 20,41 linii.

### V3

V3 wykorzystuje 11 cech planszy i 12 wag decyzyjnych. Dodano między innymi głębokość dziur, bloki nad dziurami, studnie, przejścia w wierszach i kolumnach, strefę zagrożenia oraz premię za usunięcie linii.

Końcowy agent V3 osiągnął średnio 59,864 usuniętych linii w teście końcowym na 250 nowych seedach.

## Najważniejsze pliki

| Plik | Opis |
|---|---|
| `main_v3.py` | Uruchamia finalnego agenta V3 z 11 cechami i 12 wagami. |
| `tetris_features_v3.py` | Oblicza cechy planszy i ocenę możliwych ustawień. |
| `optimize_weights_v3.py` | Prowadzi finalny trening algorytmu genetycznego. |
| `optimize_weights_v2.py` | Wcześniejszy optymalizator oparty na 4 cechach. |
| `benchmark.py` | Testuje agentów bazowych. |
| `compare_agents.py` | Porównuje wyniki różnych wersji agenta. |
| `analyze_results.py` | Wykonuje analizę statystyczną i generuje wykresy. |
| `best_weights_v3.json` | Zawiera najlepszy zestaw 12 wag V3. |
| `training_history_v3.csv` | Zawiera historię treningu V3. |
| `final_report_v3.txt` | Zawiera raport końcowy z walidacji i testu. |

Checkpoint `checkpoint_v3.pkl` znajduje się osobno w załączniku nr 7.

## Wymagania

Projekt był uruchamiany w systemie Windows z poziomu PowerShell. Wymagane biblioteki znajdują się w pliku `requirements.txt`.

## Instalacja

Przejdź do folderu z kodem źródłowym i uruchom:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Uruchomienie finalnego agenta V3

```powershell
python .\main_v3.py
```

Program wczyta `best_weights_v3.json`, utworzy środowisko z obserwacją 11 cech i rozpocznie rozgrywkę w trybie graficznym.

## Uruchomienie treningu V3

```powershell
python .\optimize_weights_v3.py
```

Jeżeli w katalogu znajduje się `checkpoint_v3.pkl`, program wznowi trening od zapisanego stanu. Jeżeli checkpointu nie ma, utworzy nową populację.

## Wyniki końcowe V3

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

## Uwagi techniczne

- Projekt nie analizuje obrazu ekranu.
- Agent nie steruje klawiszami krok po kroku.
- Środowisko generuje legalne końcowe ustawienia klocka.
- Agent wybiera ruch zachłannie i nie planuje kilku klocków do przodu.
- Algorytm genetyczny dobiera wagi przygotowanych wcześniej cech planszy.

## Autor

Adrian Rachubiński  
Kierunek: Informatyka  
Projekt dyplomowy, Poznań 2026
