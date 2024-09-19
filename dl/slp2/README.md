# Keterangan
Seluruh program berada pada kedo C++. program python untuk melakukan visualisasi data. Setelah program c++ dijalankan akan mengeluarkan output berupa combined_result_{train/test}.csv yang merupakan tahapan dari proses tersebut. Selain itu ada juga result.csv yang merupakan accuracy dan MSE untuk setiap epoch. 10 baris pertama pada result merupakan train dan 10 baris ke-2 merupakan test. 

## Cara menjalankan program
```bash
g++ -o slp slp.cpp
./slp
```
### Untuk melihat visualisasi
```bash
python plot.py
# atau jika menggunakan python3
python3 plot.py
```
