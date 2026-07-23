# Agent AI do gry w Tetris

Repozytorium zawiera załączniki do projektu dyplomowego „Optymalizacja strategii agenta grającego w Tetrisa za pomocą algorytmu genetycznego”.

## Najważniejszy wynik

Finalny agent V3 wykorzystuje 11 cech planszy i 12 wag decyzyjnych. W teście końcowym na 250 nowych seedach uzyskał średnio 59,864 usuniętych linii, medianę 55 linii i najlepszy wynik 147 linii.

## Układ repozytorium

```text
Zalaczniki_do_pracy/
├── Zalacznik_1_Kod_zrodlowy/
├── Zalacznik_2_README.md
├── Zalacznik_3_requirements.txt
├── Zalacznik_4_Najlepsze_Wagi/
├── Zalacznik_5_Dane_z_eksperymentow/
├── Zalacznik_6_Raporty_z_badan/
└── Zalacznik_7_Checkpoint_koncowego_treningu_V3/
```

## Najważniejsze pliki kodu

- `main_v3.py` – uruchamia finalnego agenta V3.
- `tetris_features_v3.py` – oblicza 11 cech planszy i ocenę kandydatów.
- `optimize_weights_v3.py` – prowadzi finalny trening algorytmu genetycznego.
- `optimize_weights_v2.py` – wcześniejsza wersja optymalizatora z 4 cechami.
- `benchmark.py` – test agenta losowego i bazowego.
- `compare_agents.py` – porównanie wersji agenta.
- `analyze_results.py` – analiza statystyczna i generowanie wykresów.

## Instalacja

Przejdź do folderu `Zalaczniki_do_pracy/Zalacznik_1_Kod_zrodlowy`, a następnie uruchom:

```powershell
python -m pip install -r requirements.txt
```

## Uruchomienie finalnego agenta V3

```powershell
python .\main_v3.py
```

Program wczytuje `best_weights_v3.json`, wykorzystuje 11 cech planszy oraz wszystkie 12 wag, w tym `line_clear_reward`.

## Uruchomienie treningu V3

```powershell
python .\optimize_weights_v3.py
```

Jeżeli w katalogu znajduje się `checkpoint_v3.pkl`, program może wznowić trening od zapisanego stanu. Checkpoint jest przekazany osobno jako załącznik nr 7.

## Wyniki finalnego V3

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

## Autor

Adrian Rachubiński  
Kierunek: Informatyka  
Projekt dyplomowy, Poznań 2026
