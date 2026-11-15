### 🎮 [▶ Play the game live](https://ab-test-tycoon.streamlit.app)
# 🎰 A/B Test Tycoon: App & Casino Designer

An interactive Streamlit game for *Introduction to Data Science (IDSE)*. You are a Product Manager with an impatient CEO; learn hypothesis testing, p-values, statistical power and p-hacking by playing.

## Levels
1. **The Obvious Winner**: real +5 pt lift; collect data, see p < 0.05, ship.
2. **The Placebo Trap**: A/A test (0 true lift). Peek too early and noise fools you (Type I error).
3. **The Subtle Gem**: real +1.5 pt lift hidden in noise; stop early and you miss it (Type II error).

## Math
Users are simulated with `np.random.binomial`; p-values use a pooled two-proportion z-test (`scipy.stats.norm`); the 95% CI uses the unpooled standard error. Nothing is hard-coded.

## Run in Google Colab
Open `AB_Test_Tycoon_Colab.ipynb` in Colab, run all 3 code cells, open the printed link (password = printed IP).

## Run locally (optional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Author
SUJAN RAM S(EC24I2008)
YASHAS M (EC24I2007)
MUKESH MG (EC24I2020)
