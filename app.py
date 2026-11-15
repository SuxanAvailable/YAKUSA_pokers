"""A/B Test Tycoon: App & Casino Designer  (IDSE mini-project)

Learn hypothesis testing, p-values, power and p-hacking by playing.
All numbers come from live simulation (np.random.binomial) + a two-proportion z-test.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

st.set_page_config(page_title="A/B Test Tycoon", page_icon="🎰", layout="wide")

ALPHA = 0.05

LEVELS = [
    dict(feature="Neon Spin Button", title="Level 1: The Obvious Winner",
         base=0.10, lift=0.05, hoped=0.05,
         brief="The CEO wants the **Neon Spin Button** live NOW. Version A = old button, "
               "Version B = neon button. Collect traffic, watch the p-value, then decide."),
    dict(feature="Double XP Pop-up", title="Level 2: The Placebo Trap",
         base=0.10, lift=0.0, hoped=0.03,
         brief="Marketing swears the **Double XP Pop-up** is a game changer. Version A and B "
               "look different... but is there a real difference? Be careful what you ship."),
    dict(feature="Golden Loot Box", title="Level 3: The Subtle Gem",
         base=0.10, lift=0.015, hoped=0.015,
         brief="The **Golden Loot Box** might help a little. The traffic is noisy and the "
               "effect is small. Can you tell a tiny real win from nothing?"),
]
BATCHES = [50, 200, 1000, 5000]


# ---------- maths ----------
def ztest(xa, na, xb, nb):
    """Two-proportion z-test. Returns (z, two-tailed p, 95% CI of B-A)."""
    if na == 0 or nb == 0:
        return 0.0, 1.0, (0.0, 0.0)
    pa, pb = xa / na, xb / nb
    pool = (xa + xb) / (na + nb)
    se = np.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
    z = 0.0 if se == 0 else (pb - pa) / se
    p = float(2 * stats.norm.sf(abs(z)))
    se_d = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = pb - pa
    return float(z), p, (d - 1.96 * se_d, d + 1.96 * se_d)


def power(n, base, delta):
    """Approx. power of a two-sided z-test (alpha=0.05) to detect lift `delta` with n per variant."""
    if n <= 0 or delta == 0:
        return ALPHA
    p1 = base + delta
    se = np.sqrt(base * (1 - base) / n + p1 * (1 - p1) / n)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    return float(stats.norm.sf(zc - abs(delta) / se) + stats.norm.cdf(-zc - abs(delta) / se))


# ---------- state ----------
def reset_level(i):
    ss = st.session_state
    ss.level = i
    ss.na = ss.nb = ss.xa = ss.xb = 0
    ss.hist = []      # (n, rate A, rate B, p)
    ss.result = None
    ss.msg = ""


def reset_game():
    st.session_state.cash = 100_000
    st.session_state.rep = 50
    reset_level(0)


if "level" not in st.session_state:
    reset_game()
ss = st.session_state


def stats_now():
    return ztest(ss.xa, ss.na, ss.xb, ss.nb)


# ---------- callbacks ----------
def run_batch(n):
    L = LEVELS[ss.level]
    ss.xa += int(np.random.binomial(n, L["base"]))
    ss.xb += int(np.random.binomial(n, L["base"] + L["lift"]))
    ss.na += n
    ss.nb += n
    _, p, _ = stats_now()
    ss.hist.append((ss.na, ss.xa / ss.na, ss.xb / ss.nb, p))
    ss.msg = ""


def finish(kind, title, text, dcash, drep):
    ss.cash += dcash
    ss.rep = max(0, min(100, ss.rep + drep))
    ss.result = dict(kind=kind, title=title, text=text, dcash=dcash, drep=drep)


def decide(ship):
    i, n = ss.level, ss.na
    _, p, _ = stats_now()
    sig = p < ALPHA and ss.xb / n > ss.xa / n
    if i == 0:
        if ship and sig:
            finish("good", "Shipped with evidence! 🚀",
                   f"p = {p:.4f} < 0.05 and the lift is real. Players love the neon button. "
                   "That's how a clean experiment feels.", 60_000, 10)
        elif ship:
            finish("meh", "Lucky guess 🍀",
                   f"You shipped at p = {p:.3f} (not significant). The button WAS genuinely better, "
                   "but you had no evidence. Next time, you might not be so lucky.", 25_000, -5)
        else:
            finish("bad", "You rejected a real winner 😬",
                   "The neon button truly was +5 points better. Rejecting it is a Type II error "
                   "(a missed discovery).", -20_000, -10)
    elif i == 1:
        if ship:
            if p < ALPHA:
                finish("bad", "FALSE POSITIVE! (Type I error) 🤡",
                       f"You shipped at p = {p:.3f}. But A and B were secretly IDENTICAL. "
                       "You peeked, and random noise dipped below 0.05 by luck. "
                       "Checking after every batch gives noise many chances to fool you (p-hacking).",
                       -60_000, -25)
            else:
                finish("bad", "Shipped a placebo 💸",
                       f"p = {p:.2f} was not even significant, and the feature does nothing. "
                       "Dev time wasted.", -40_000, -15)
        elif n < 1000:
            ss.msg = ("You can't claim 'no effect' from so few users. Collect at least "
                      "1,000 users per variant first.")
        else:
            extra = (f" (p = {p:.3f} is below 0.05 right now: that is the 5% false-alarm rate "
                     "built into alpha. The CI shows how tiny any effect is.)") if p < ALPHA else ""
            finish("good", "Placebo rejected! 🕵️",
                   f"With {n:,} users per variant the two curves converged: no real lift. "
                   "You saved the company a useless launch." + extra, 30_000, 15)
    else:
        if ship:
            if sig:
                finish("good", "Subtle gem shipped! 💎",
                       f"p = {p:.4f} with {n:,} users per variant. A +1.5 point lift is small but "
                       "adds up over millions of spins. Big sample, big reveal.", 120_000, 20)
            else:
                finish("meh", "Lucky again 🍀",
                       f"You shipped at p = {p:.3f}. The gem was real, but your evidence was weak.",
                       40_000, 0)
        else:
            if n < 3000:
                text = (f"You stopped at {n:,} users per variant. The effect is small, so noise hid "
                        "it. Needed roughly 6,500+ per variant for 80% power. That's a Type II error.")
            else:
                text = (f"Even with {n:,} users the test was still short of what it needed "
                        "(about 6,500+ per variant for 80% power). Underpowered tests miss small real "
                        "effects: a Type II error.")
            finish("bad", "You buried a gem 😭", text, -50_000, -10)


def go_next():
    reset_level(ss.level + 1)


# ---------- CEO ----------
def ceo_line():
    if ss.na == 0:
        return random.choice([
            "Why isn't this live yet?! Click something!",
            "I have a golf game at 3. Results by lunch, please.",
            "Just add users. The more clicks the more money. That's math."])
    _, p, _ = stats_now()
    better = ss.xb / ss.na > ss.xa / ss.na
    if p < ALPHA and better:
        return random.choice([
            f"p = {p:.3f}, that's UNDER 0.05! It's SCIENCE! SHIP IT NOW!!!",
            "I see a green number. Why are you still staring at the screen? SHIP. IT.",
            "Significant means done. Press the rocket button!"])
    if ss.level == 2:
        return random.choice([
            "Not significant? Then it's useless. Kill it and move on!",
            "Your chart is boring. Cancel it before I lose interest."])
    if ss.na < 500:
        return random.choice([
            f"Only {ss.na} users?? My cat gets more traffic. Add more!",
            "Is it done yet? How about now? Now?"])
    return random.choice([
        f"p = {p:.2f} is bigger than 0.05, right? That's... bad? Make it smaller.",
        "Can we just round the p-value down? Asking for a friend."])


# ---------- chart ----------
def draw_chart():
    h = ss.hist
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    if h:
        n = [x[0] for x in h]
        a1.plot(n, [x[1] * 100 for x in h], marker="o", ms=3, label="A (control)")
        a1.plot(n, [x[2] * 100 for x in h], marker="o", ms=3, label="B (new feature)")
        a2.plot(n, [x[3] for x in h], color="purple", marker="o", ms=3, label="p-value")
        a1.legend()
    a1.set_ylabel("Conversion rate (%)")
    a1.set_title("Conversion rates over time")
    a1.grid(alpha=0.3)
    a2.axhline(ALPHA, color="red", ls="--", label="p = 0.05")
    a2.axhspan(0, ALPHA, color="red", alpha=0.08)
    a2.set_ylim(0, 1)
    a2.set_ylabel("p-value")
    a2.set_xlabel("Users per variant")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------- UI ----------
st.title("🎰 A/B Test Tycoon: App & Casino Designer")

with st.expander("📖 How to Play & Learning Objectives"):
    st.markdown("""
**You are the Lead Product Manager at a mobile gaming studio.** Your impatient CEO wants features shipped fast.

**How to play**
1. Press a **+users** button to simulate traffic (users per variant, A and B).
2. Watch the conversion chart, the **p-value** and the **power** meter.
3. Press **Ship B** or **Reject B**. Money and reputation change depending on whether you were right.
4. Survive 3 levels.

**Learning objectives**
- **Hypothesis testing:** H0 = A and B convert equally; reject H0 only with evidence.
- **p-value (< 0.05):** how surprising the data is if H0 were true. It is NOT the chance B is better.
- **Power & sample size:** small samples = noisy rates, and small real effects need many users.
- **P-hacking / peeking:** checking after every batch and stopping when p < 0.05 inflates false positives (Type I).
- **Type I vs Type II errors:** shipping a placebo vs burying a real gem.
""")

if ss.level >= 3:
    st.header("🏁 Game Over")
    c1, c2 = st.columns(2)
    c1.metric("Company cash", f"${ss.cash:,}")
    c2.metric("Reputation", f"{ss.rep}/100")
    st.markdown("""
**Takeaways**
- Don't stop a test the first time p < 0.05: decide your sample size first.
- No significant result with a small sample is not proof of "no effect" (low power).
- Bigger samples make noisy rates settle and make small real effects visible.
""")
    st.button("🔄 Play again", on_click=reset_game)
    st.stop()

L = LEVELS[ss.level]
m1, m2, m3 = st.columns(3)
m1.metric("💰 Company cash", f"${ss.cash:,}")
m2.metric("⭐ Reputation", f"{ss.rep}/100")
m3.metric("Level", f"{ss.level + 1}/3")

st.subheader(L["title"])
st.info(L["brief"])
if ss.result is None:
    st.warning(f"🧑‍💼 **CEO:** {ceo_line()}")

left, right = st.columns([3, 2])
with left:
    draw_chart()

with right:
    st.markdown(f"**Traffic so far:** {ss.na:,} users per variant")
    bcols = st.columns(len(BATCHES))
    for col, n in zip(bcols, BATCHES):
        col.button(f"+{n:,}", key=f"b{n}", on_click=run_batch, args=(n,),
                   disabled=ss.result is not None, use_container_width=True,
                   help="users added to each variant")

    z, p, (lo, hi) = stats_now()
    ra = ss.xa / ss.na if ss.na else 0.0
    rb = ss.xb / ss.na if ss.na else 0.0
    c1, c2 = st.columns(2)
    c1.metric("A rate", f"{ra * 100:.2f}%")
    c2.metric("B rate", f"{rb * 100:.2f}%", f"{(rb - ra) * 100:+.2f} pts")

    st.metric("P-value (two-tailed)", f"{p:.4f}")
    st.progress(float(min(p, 1.0)))
    st.caption("🟢 Significant (p < 0.05)" if p < ALPHA and ss.na else "🔴 Not significant (p ≥ 0.05)")
    st.caption(f"95% CI of lift (B − A): [{lo * 100:+.2f}, {hi * 100:+.2f}] pts")

    pw = power(ss.na, L["base"], L["hoped"])
    st.progress(float(min(pw, 1.0)))
    st.caption(f"Power to detect a {L['hoped'] * 100:.1f}-pt lift at n = {ss.na:,}: "
               f"{pw * 100:.0f}% (aim for ≥ 80%)")

    if ss.level == 1 and ss.hist:
        dips = sum(1 for h in ss.hist if h[3] < ALPHA)
        st.caption(f"⚠️ Peeks where p < 0.05: {dips} of {len(ss.hist)}")

    if ss.msg:
        st.warning(ss.msg)

    if ss.result is None:
        d1, d2 = st.columns(2)
        d1.button("🚀 Ship B", on_click=decide, args=(True,),
                  disabled=ss.na == 0, use_container_width=True)
        d2.button("🗑️ Reject B", on_click=decide, args=(False,),
                  disabled=ss.na == 0, use_container_width=True)
    else:
        r = ss.result
        box = {"good": st.success, "bad": st.error, "meh": st.info}[r["kind"]]
        box(f"**{r['title']}**\n\n{r['text']}\n\nCash {r['dcash']:+,} | Reputation {r['drep']:+d}")
        st.button("Next ➡️" if ss.level < 2 else "🏁 Final score", on_click=go_next)
