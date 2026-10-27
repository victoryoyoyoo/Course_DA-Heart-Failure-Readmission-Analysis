# 心臟衰竭再入院預測

用 3000 筆心臟衰竭病患資料，預測 30 天內會不會再入院，也預測 BNP 數值。

資料集是課程提供的，第一次執行會自動下載成 `D01_heart_failure_readmission_dataset.csv`。

## 怎麼跑

需要 Python 3.12（requirements.txt 的版本是用 3.12 裝的）。

```bash
pip install -r requirements.txt
python Heart-Failure-Readmission-Analysis.py
```

跑的時候會跳出好幾個圖表視窗，要關掉目前這個視窗，程式才會繼續往下跑。

資料先切成訓練集和測試集，補值、離群值處理、縮放、特徵選擇都只用訓練集學，測試集只做轉換。
