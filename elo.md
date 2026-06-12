PS C:\Programming\Python\CorazonesRL> .venv\Scripts\Activate.ps1; python -m src.elo_torneo --partidas 30 --min-paso 10400000 --max-snapshots 12 --elo-puro
============================================================

🏆 Torneo Elo — 12 snapshots (v5) | Modo: Puro (adyacentes)
   Partidas por enfrentamiento: 30
   Total de partidas: 1980
============================================================

  [1/66] snapshot_0010400000 vs snapshot_0010500000 ... 20-10
  [2/66] snapshot_0010400000 vs snapshot_0014500000 ... 19-11
  [3/66] snapshot_0010400000 vs snapshot_0014600000 ... 19-11
  [4/66] snapshot_0010400000 vs snapshot_0014700000 ... 16-14
  [5/66] snapshot_0010400000 vs snapshot_0014800000 ... 18-11
  [6/66] snapshot_0010400000 vs snapshot_0014900000 ... 17-12
  [7/66] snapshot_0010400000 vs snapshot_0015000000 ... 19-11
  [8/66] snapshot_0010400000 vs snapshot_0015100000 ... 17-13
  [9/66] snapshot_0010400000 vs snapshot_0015200000 ... 19-11
  [10/66] snapshot_0010400000 vs snapshot_0015300000 ... 17-10
  [11/66] snapshot_0010400000 vs snapshot_0015400000 ... 18-12
  [12/66] snapshot_0010500000 vs snapshot_0014500000 ... 17-13
  [13/66] snapshot_0010500000 vs snapshot_0014600000 ... 18-12
  [14/66] snapshot_0010500000 vs snapshot_0014700000 ... 17-13
  [15/66] snapshot_0010500000 vs snapshot_0014800000 ... 21-9
  [16/66] snapshot_0010500000 vs snapshot_0014900000 ... 19-10
  [17/66] snapshot_0010500000 vs snapshot_0015000000 ... 22-7
  [18/66] snapshot_0010500000 vs snapshot_0015100000 ... 19-11
  [19/66] snapshot_0010500000 vs snapshot_0015200000 ... 18-11
  [20/66] snapshot_0010500000 vs snapshot_0015300000 ... 20-10
  [21/66] snapshot_0010500000 vs snapshot_0015400000 ... 21-8
  [22/66] snapshot_0014500000 vs snapshot_0014600000 ... 20-9
  [23/66] snapshot_0014500000 vs snapshot_0014700000 ... 19-11
  [24/66] snapshot_0014500000 vs snapshot_0014800000 ... 21-8
  [25/66] snapshot_0014500000 vs snapshot_0014900000 ... 17-13
  [26/66] snapshot_0014500000 vs snapshot_0015000000 ... 16-13
  [27/66] snapshot_0014500000 vs snapshot_0015100000 ... 20-10
  [28/66] snapshot_0014500000 vs snapshot_0015200000 ... 19-11
  [29/66] snapshot_0014500000 vs snapshot_0015300000 ... 22-8
  [30/66] snapshot_0014500000 vs snapshot_0015400000 ... 17-13
  [31/66] snapshot_0014600000 vs snapshot_0014700000 ... 19-11
  [32/66] snapshot_0014600000 vs snapshot_0014800000 ... 21-9
  [33/66] snapshot_0014600000 vs snapshot_0014900000 ... 18-12
  [34/66] snapshot_0014600000 vs snapshot_0015000000 ... 18-11
  [35/66] snapshot_0014600000 vs snapshot_0015100000 ... 19-11
  [36/66] snapshot_0014600000 vs snapshot_0015200000 ... 21-9
  [37/66] snapshot_0014600000 vs snapshot_0015300000 ... 20-10
  [38/66] snapshot_0014600000 vs snapshot_0015400000 ... 18-11
  [39/66] snapshot_0014700000 vs snapshot_0014800000 ... 19-11
  [40/66] snapshot_0014700000 vs snapshot_0014900000 ... 16-14
  [41/66] snapshot_0014700000 vs snapshot_0015000000 ... 15-15
  [42/66] snapshot_0014700000 vs snapshot_0015100000 ... 19-11
  [43/66] snapshot_0014700000 vs snapshot_0015200000 ... 20-10
  [44/66] snapshot_0014700000 vs snapshot_0015300000 ... 20-10
  [45/66] snapshot_0014700000 vs snapshot_0015400000 ... 15-15
  [46/66] snapshot_0014800000 vs snapshot_0014900000 ... 18-12
  [47/66] snapshot_0014800000 vs snapshot_0015000000 ... 15-15
  [48/66] snapshot_0014800000 vs snapshot_0015100000 ... 18-12
  [49/66] snapshot_0014800000 vs snapshot_0015200000 ... 17-13
  [50/66] snapshot_0014800000 vs snapshot_0015300000 ... 16-14
  [51/66] snapshot_0014800000 vs snapshot_0015400000 ... 14-16
  [52/66] snapshot_0014900000 vs snapshot_0015000000 ... 17-13
  [53/66] snapshot_0014900000 vs snapshot_0015100000 ... 18-12
  [54/66] snapshot_0014900000 vs snapshot_0015200000 ... 18-12
  [55/66] snapshot_0014900000 vs snapshot_0015300000 ... 18-12
  [56/66] snapshot_0014900000 vs snapshot_0015400000 ... 16-14
  [57/66] snapshot_0015000000 vs snapshot_0015100000 ... 19-10
  [58/66] snapshot_0015000000 vs snapshot_0015200000 ... 22-8
  [59/66] snapshot_0015000000 vs snapshot_0015300000 ... 22-8
  [60/66] snapshot_0015000000 vs snapshot_0015400000 ... 18-12
  [61/66] snapshot_0015100000 vs snapshot_0015200000 ... 19-11
  [62/66] snapshot_0015100000 vs snapshot_0015300000 ... 15-15
  [63/66] snapshot_0015100000 vs snapshot_0015400000 ... 14-15
  [64/66] snapshot_0015200000 vs snapshot_0015300000 ... 15-15
  [65/66] snapshot_0015200000 vs snapshot_0015400000 ... 14-16
  [66/66] snapshot_0015300000 vs snapshot_0015400000 ... 18-12

============================================================
🏆 CLASIFICACIÓN FINAL
============================================================

Pos   Snapshot                       Paso         Elo      Δ desde anterior
------------------------------------------------------------

1     snapshot_0015400000            15,400,000  1900
2     snapshot_0015300000            15,300,000  1770     -130
3     snapshot_0015200000            15,200,000  1583     -187
4     snapshot_0015000000            15,000,000  1548     -35
5     snapshot_0015100000            15,100,000  1512     -37
6     snapshot_0010500000            10,500,000  1436     -76
7     snapshot_0014900000            14,900,000  1415     -20
8     snapshot_0014600000            14,600,000  1402     -13
9     snapshot_0014500000            14,500,000  1368     -35
10    snapshot_0010400000            10,400,000  1364     -4
11    snapshot_0014700000            14,700,000  1360     -4
12    snapshot_0014800000            14,800,000  1342     -18
(.venv) PS C:\Programming\Python\CorazonesRL>
