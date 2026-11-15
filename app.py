"""A/B Test Tycoon: App & Casino Designer (IDSE mini-project)
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
BATCHES = [50, 200, 1000, 5000]
LEVELS = [
    dict(feature="Neon Spin Button", icon="🔘", title="Level 1: The Obvious Winner",
         base=0.10, lift=0.05, hoped=0.05,
         brief="The CEO wants the **Neon Spin Button** live NOW. A = old button, B = neon button."),
    dict(feature="Double XP Pop-up", icon="✨", title="Level 2: The Placebo Trap",
         base=0.10, lift=0.0, hoped=0.03,
         brief="Marketing swears the **Double XP Pop-up** is a game changer. Is it? Careful what you ship."),
    dict(feature="Golden Loot Box", icon="💎", title="Level 3: The Subtle Gem",
         base=0.10, lift=0.015, hoped=0.015,
         brief="The **Golden Loot Box** might help a little. Traffic is noisy. Can you spot a small real win?"),
]

st.markdown("""<style>
.stApp{background:radial-gradient(circle at top,#1d1140,#0b0b16 60%)}
h1,h2,h3{color:#ffd84d!important}
.hud{display:flex;gap:10px;margin:4px 0 12px}
.hud div{flex:1;background:#1c1738;border:1px solid #6c3cff;border-radius:14px;padding:8px 12px;text-align:center;box-shadow:0 0 12px #6c3cff55}
.hud b{display:block;font-size:1.5rem;color:#ffd84d}.hud span{font-size:.7rem;color:#a99cff;letter-spacing:.1em}
.ceo{background:#2a1530;border:2px solid #ff4d8d;border-radius:16px;padding:10px 16px;font-size:1.05rem;color:#fff}
.coach{background:#10243a;border-left:5px solid #4dd8ff;border-radius:10px;padding:10px 14px;color:#dff6ff}
.lbl{font-size:.8rem;color:#a99cff;margin:8px 0 2px}
.bar{background:#241f45;border-radius:10px;height:18px;position:relative;overflow:hidden}
.bar i{display:block;height:100%;border-radius:10px}
.mark{position:absolute;top:0;bottom:0;width:3px;background:#fff}
.dots{font-size:.85rem;line-height:1.15;letter-spacing:1px}
.big{font-size:1.6rem;font-weight:700;color:#fff}
</style>""", unsafe_allow_html=True)


# ---------- maths ----------
def ztest(xa, na, xb, nb):
    """Pooled two-proportion z-test -> (z, two-tailed p, 95% CI of B-A)."""
    if na == 0 or nb == 0:
        return 0.0, 1.0, (0.0, 0.0)
    pa, pb = xa / na, xb / nb
    pool = (xa + xb) / (na + nb)
    se = np.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
    z = 0.0 if se == 0 else (pb - pa) / se
    se_d = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return float(z), float(2 * stats.norm.sf(abs(z))), (pb - pa - 1.96 * se_d, pb - pa + 1.96 * se_d)


def power(n, base, delta):
    if n <= 0 or delta == 0:
        return ALPHA
    p1 = base + delta
    se = np.sqrt(base * (1 - base) / n + p1 * (1 - p1) / n)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    return float(stats.norm.sf(zc - abs(delta) / se) + stats.norm.cdf(-zc - abs(delta) / se))


def n_needed(base, delta, pw=0.8):
    p1 = base + delta
    return int(np.ceil((stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(pw)) ** 2
                       * (base * (1 - base) + p1 * (1 - p1)) / delta ** 2))


# ---------- state ----------
def reset_level(i):
    ss = st.session_state
    ss.level, ss.na, ss.xa, ss.xb = i, 0, 0, 0
    ss.hist, ss.result, ss.msg, ss.last = [], None, "", None


def reset_game():
    st.session_state.cash, st.session_state.rep, st.session_state.badges = 100_000, 50, []
    reset_level(0)


if "level" not in st.session_state:
    reset_game()
ss = st.session_state


def cur():
    return ztest(ss.xa, ss.na, ss.xb, ss.na)


def run_batch(n):
    L = LEVELS[ss.level]
    ca, cb = int(np.random.binomial(n, L["base"])), int(np.random.binomial(n, L["base"] + L["lift"]))
    ss.xa, ss.xb, ss.na = ss.xa + ca, ss.xb + cb, ss.na + n
    ss.last = (n, ca, cb)
    _, p, _ = cur()
    if ss.level == 1 and ss.hist and ss.hist[-1][3] >= ALPHA > p:
        st.toast("🚨 p just dipped below 0.05... real, or luck?", icon="👀")
    ss.hist.append((ss.na, ss.xa / ss.na, ss.xb / ss.na, p))
    ss.msg = ""


def finish(kind, title, text, dcash, drep, badge=None):
    ss.cash += dcash
    ss.rep = max(0, min(100, ss.rep + drep))
    if badge and badge not in ss.badges:
        ss.badges.append(badge)
    ss.result = dict(kind=kind, title=title, text=text, dcash=dcash, drep=drep)


def decide(ship):
    i, n = ss.level, ss.na
    _, p, _ = cur()
    sig = p < ALPHA and ss.xb > ss.xa
    if i == 0:
        if ship and sig:
            finish("good", "Shipped with evidence! 🚀", f"p = {p:.4f} < 0.05 and the lift is real. Players love it.", 60_000, 10, "🚀 Data-Driven")
        elif ship:
            finish("meh", "Lucky guess 🍀", f"You shipped at p = {p:.3f} (not significant). It WAS better, but you had no evidence.", 25_000, -5)
        else:
            finish("bad", "You rejected a real winner 😬", "The neon button truly was +5 pts better. Rejecting a real effect = Type II error.", -20_000, -10)
    elif i == 1:
        if ship and p < ALPHA:
            finish("bad", "FALSE POSITIVE! (Type I) 🤡", f"You shipped at p = {p:.3f}, but A and B were IDENTICAL. Every peek is another lottery ticket, and noise won one (p-hacking).", -60_000, -25, "🤡 Peek Victim")
        elif ship:
            finish("bad", "Shipped a placebo 💸", f"p = {p:.2f} wasn't even significant, and the feature does nothing.", -40_000, -15)
        elif n < 1000:
            ss.msg = "Can't claim 'no effect' from so few users. Collect at least 1,000 per variant first."
        else:
            extra = f" (p = {p:.3f} is under 0.05 right now: that's the 5% false-alarm rate built into alpha.)" if p < ALPHA else ""
            finish("good", "Placebo rejected! 🕵️", f"With {n:,} users per variant the curves converged: no real lift." + extra, 30_000, 15, "🧘 Patient Scientist")
    else:
        if ship and sig:
            finish("good", "Subtle gem shipped! 💎", f"p = {p:.4f} with {n:,} users per variant. Small lift, big sample, clear signal.", 120_000, 20, "💎 Gem Finder")
        elif ship:
            finish("meh", "Lucky again 🍀", f"Shipped at p = {p:.3f}. The gem was real but your evidence was weak.", 40_000, 0)
        else:
            t = (f"You stopped at {n:,}/variant. Small effects hide in noise." if n < 3000
                 else f"Even {n:,}/variant fell short.") + f" You needed about {n_needed(0.10, 0.015):,} per variant for 80% power. Type II error."
            finish("bad", "You buried a gem 😭", t, -50_000, -10)


def go_next():
    reset_level(ss.level + 1)


# ---------- helpers ----------
def bar(frac, color, mark=None):
    m = f'<div class="mark" style="left:{mark * 100:.1f}%"></div>' if mark is not None else ""
    return f'<div class="bar"><i style="width:{max(0, min(1, frac)) * 100:.1f}%;background:{color}"></i>{m}</div>'


def dots(rate, total=60):
    g = round(rate * total)
    s = "🟩" * g + "⬛" * (total - g)
    return "<br>".join(s[k:k + 20] for k in range(0, len(s), 20))


def ceo():
    if ss.na == 0:
        return "😎", random.choice(["Why isn't this live yet?! Click something!", "Golf at 3. Results by lunch, please."])
    _, p, _ = cur()
    if p < ALPHA and ss.xb > ss.xa:
        return "🤑", random.choice([f"p = {p:.3f}, UNDER 0.05! It's SCIENCE! SHIP IT NOW!!!", "I see a green number. SHIP. IT.", "Significant means done. Press the rocket!"])
    if ss.level == 2:
        return "😤", random.choice(["Not significant? Useless. Kill it!", "Your chart is boring. Cancel it."])
    if ss.na < 500:
        return "😤", random.choice([f"Only {ss.na} users?? My cat gets more traffic.", "Is it done yet? Now? Now?"])
    return "😤", random.choice([f"p = {p:.2f} is bigger than 0.05... that's bad? Make it smaller.", "Can we just round the p-value down?"])


def coach():
    if ss.na == 0:
        return "**Hypothesis:** H0 = 'A and B convert equally'. You only reject H0 if the data would be very surprising under it (p < 0.05). Add traffic to start."
    _, p, (lo, hi) = cur()
    k = sum(1 for h in ss.hist if h[3] < ALPHA)
    if ss.na < 300:
        return "Small sample: rates jump around and the shaded bands (95% CI) are wide. Noise can look like a result."
    if ss.level == 1 and p < ALPHA:
        return f"p < 0.05 right now, but you've looked {len(ss.hist)} times. More looks = more chances for noise to fool you. Decide your sample size BEFORE peeking."
    if lo < 0 < hi:
        return "The CI of the lift includes 0, so 'no difference' is still plausible. More data narrows the bands."
    return "The CI of the lift excludes 0 and p < 0.05: strong evidence. Bigger samples make this reliable."


def dark_fig(rows):
    fig, axs = plt.subplots(rows, 1, figsize=(7, 3 * rows), sharex=rows > 1)
    fig.patch.set_facecolor("#0b0b16")
    for a in np.atleast_1d(axs):
        a.set_facecolor("#14122a")
        a.tick_params(colors="white")
        a.grid(alpha=0.2)
        for s in a.spines.values():
            s.set_color("#6c3cff")
        a.yaxis.label.set_color("white"); a.xaxis.label.set_color("white"); a.title.set_color("white")
    return fig, axs


def draw_chart(reveal=None):
    h = ss.hist
    fig, (a1, a2) = dark_fig(2)
    if h:
        n = np.array([x[0] for x in h]); ra = np.array([x[1] for x in h]); rb = np.array([x[2] for x in h])
        for r, c, lab in ((ra, "#4dd8ff", "A (control)"), (rb, "#ff4d8d", "B (new feature)")):
            ci = 1.96 * np.sqrt(r * (1 - r) / n)
            a1.plot(n, r * 100, color=c, marker="o", ms=3, label=lab)
            a1.fill_between(n, (r - ci) * 100, (r + ci) * 100, color=c, alpha=0.18)
        a2.plot(n, [x[3] for x in h], color="#c89bff", marker="o", ms=3)
        a1.legend(facecolor="#14122a", labelcolor="white")
    if reveal:
        a1.axhline(reveal[0] * 100, color="gold", ls="--", label="true A")
        a1.axhline(reveal[1] * 100, color="orange", ls=":", label="true B")
    a1.set_ylabel("Conversion (%) ± 95% CI"); a1.set_title("Rates converge as users grow")
    a2.axhline(ALPHA, color="red", ls="--"); a2.axhspan(0, ALPHA, color="red", alpha=0.12)
    a2.text(0.01, 0.07, "significant zone (p<0.05)", color="#ff8a8a", transform=a2.transAxes, fontsize=8)
    a2.set_ylim(0, 1); a2.set_ylabel("p-value"); a2.set_xlabel("Users per variant")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ---------- tabs ----------
def final():
    st.header("🏁 Game Over")
    st.markdown(f'<div class="hud"><div><b>${ss.cash:,}</b><span>FINAL CASH</span></div>'
                f'<div><b>{ss.rep}/100</b><span>REPUTATION</span></div>'
                f'<div><b>{len(ss.badges)}/4</b><span>BADGES</span></div></div>', unsafe_allow_html=True)
    st.write(" ".join(f"`{b}`" for b in ss.badges) or "No badges yet. Try again!")
    st.markdown("**Takeaways:** fix your sample size before peeking; 'not significant' with few users ≠ 'no effect'; bigger samples calm noisy rates and reveal small real effects. Try the 🧪 Lab tab!")
    st.button("🔄 Play again", on_click=reset_game)


def play_tab():
    if ss.level >= 3:
        return final()
    L = LEVELS[ss.level]
    st.markdown(f'<div class="hud"><div><b>${ss.cash:,}</b><span>💰 COMPANY CASH</span></div>'
                f'<div><b>{ss.rep}/100</b><span>⭐ REPUTATION</span></div>'
                f'<div><b>{ss.level + 1}/3</b><span>LEVEL</span></div>'
                f'<div><b>{len(ss.badges)}</b><span>🏅 BADGES</span></div></div>', unsafe_allow_html=True)
    st.subheader(f"{L['icon']} {L['title']}")
    st.markdown(L["brief"])
    if ss.result is None:
        face, line = ceo()
        st.markdown(f'<div class="ceo">{face} <b>CEO:</b> {line}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="coach" style="margin-top:8px">🧠 <b>Coach:</b> {coach()}</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        rv = (L["base"], L["base"] + L["lift"]) if ss.result else None
        draw_chart(rv)
        if ss.last:
            n, ca, cb = ss.last
            st.caption(f"Latest batch: +{n:,} users per variant (60 sampled users shown) 🟩 converted ⬛ didn't")
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="dots">A<br>{dots(ca / n)}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="dots">B<br>{dots(cb / n)}</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f"**Traffic so far:** `{ss.na:,}` users per variant")
        for col, n in zip(st.columns(4), BATCHES):
            col.button(f"➕{n:,}", key=f"b{n}", on_click=run_batch, args=(n,), disabled=ss.result is not None, use_container_width=True)
        _, p, (lo, hi) = cur()
        ra, rb = (ss.xa / ss.na, ss.xb / ss.na) if ss.na else (0.0, 0.0)
        st.markdown(f'<div class="hud"><div><b>{ra * 100:.2f}%</b><span>A RATE</span></div>'
                    f'<div><b>{rb * 100:.2f}%</b><span>B RATE ({(rb - ra) * 100:+.2f})</span></div></div>', unsafe_allow_html=True)
        ev = min(1.0, -np.log10(max(p, 1e-4)) / 4)
        st.markdown(f'<div class="lbl">🎯 Evidence meter (p = {p:.4f}). Cross the white line = significant</div>'
                    + bar(ev, "#3ddc84" if p < ALPHA and ss.na else "#ff5c5c", -np.log10(ALPHA) / 4), unsafe_allow_html=True)
        pw = power(ss.na, L["base"], L["hoped"])
        st.markdown(f'<div class="lbl">🔋 Power to detect a {L["hoped"] * 100:.1f}-pt lift: {pw * 100:.0f}% (aim ≥ 80%)</div>'
                    + bar(pw, "#3ddc84" if pw >= 0.8 else "#ffb347", 0.8), unsafe_allow_html=True)
        st.caption(f"95% CI of lift (B−A): [{lo * 100:+.2f}, {hi * 100:+.2f}] pts")
        if ss.level == 1 and ss.hist:
            st.caption(f"👀 Peeks where p < 0.05: {sum(1 for h in ss.hist if h[3] < ALPHA)} of {len(ss.hist)}")
        if ss.msg:
            st.warning(ss.msg)
        if ss.result is None:
            d1, d2 = st.columns(2)
            d1.button("🚀 Ship B", on_click=decide, args=(True,), disabled=ss.na == 0, use_container_width=True)
            d2.button("🗑️ Reject B", on_click=decide, args=(False,), disabled=ss.na == 0, use_container_width=True)
            st.button("↩️ Restart level", on_click=reset_level, args=(ss.level,))
        else:
            r = ss.result
            {"good": st.success, "bad": st.error, "meh": st.info}[r["kind"]](
                f"**{r['title']}**\n\n{r['text']}\n\nCash {r['dcash']:+,} | Reputation {r['drep']:+d}")
            st.caption(f"🔍 Truth revealed: A = {L['base'] * 100:.1f}%, B = {(L['base'] + L['lift']) * 100:.1f}% (dashed lines on chart)")
            st.button("Next ➡️" if ss.level < 2 else "🏁 Final score", on_click=go_next)


def lab_tab():
    st.markdown("### 🧪 Peeking Lab: see p-hacking happen")
    st.write("500 fake **A/A tests** (no real difference, 10% vs 10%). Compare two strategies for analysing them.")
    steps = st.select_slider("Peek after every ... users", [25, 50, 100, 250], value=50)
    if st.button("▶ Run 500 experiments"):
        rng = np.random.default_rng()
        k = 1000 // steps
        a = rng.binomial(steps, 0.1, (500, k)).cumsum(1)
        b = rng.binomial(steps, 0.1, (500, k)).cumsum(1)
        n = steps * np.arange(1, k + 1)
        pool = (a + b) / (2 * n)
        se = np.sqrt(np.maximum(pool * (1 - pool) * 2 / n, 1e-12))
        p = 2 * stats.norm.sf(np.abs((b - a) / n / se))
        hit = (p < ALPHA).any(1)
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="hud"><div><b>{hit.mean() * 100:.0f}%</b><span>FALSE POSITIVES IF YOU PEEK & STOP AT FIRST p&lt;0.05</span></div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="hud"><div><b>{(p[:, -1] < ALPHA).mean() * 100:.0f}%</b><span>FALSE POSITIVES IF YOU LOOK ONCE AT n=1000</span></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig(1)
        for j in range(40):
            ax.plot(n, p[j], color="#ff5c5c" if hit[j] else "#8a86b8", alpha=0.8 if hit[j] else 0.35, lw=1)
        ax.axhline(ALPHA, color="white", ls="--"); ax.set_ylim(0, 1)
        ax.set_xlabel("Users per variant"); ax.set_ylabel("p-value")
        ax.set_title("40 A/A experiments: red = dipped below 0.05 at some point")
        st.pyplot(fig); plt.close(fig)
        st.info("Nothing real exists here, yet peeking 'finds' effects far more than 5% of the time. Looking once at a fixed sample size keeps false alarms near 5%.")
    st.markdown("### 📏 Sample-size lab: why big n calms the noise")
    n = st.slider("Users per variant", 20, 5000, 100, step=20)
    rates = np.random.binomial(n, 0.10, 400) / n
    fig, ax = dark_fig(1)
    ax.hist(rates * 100, bins=np.linspace(0, 25, 60), color="#4dd8ff")
    ax.axvline(10, color="gold", ls="--"); ax.set_xlim(0, 25)
    ax.set_xlabel("Observed conversion rate (%) (true = 10%)"); ax.set_ylabel("# of 400 experiments")
    st.pyplot(fig); plt.close(fig)
    st.caption(f"Standard error = √(p(1−p)/n) = {np.sqrt(0.09 / n) * 100:.2f} pts. 4× the users → half the noise.")
    st.markdown("### 🧮 Sample-size calculator")
    d = st.slider("True lift to detect (percentage points)", 0.5, 8.0, 1.5, step=0.5) / 100
    st.write(f"To detect a **{d * 100:.1f}-pt** lift (10% → {(0.1 + d) * 100:.1f}%) with 80% power you need about **{n_needed(0.10, d):,} users per variant**. Halve the lift → ~4× the users.")


def learn_tab():
    st.markdown("""### 📚 Concept cards
**Hypothesis test.** H0: 'no difference'. H1: 'B is different'. We keep H0 unless data are very surprising under it.

**p-value.** If H0 were true, how often would we see a gap at least this big by pure luck? p < 0.05 means 'rare enough to doubt H0'. It is **not** the probability that B is better.

**Type I error (false positive).** Shipping a feature that does nothing. With alpha = 0.05 you accept ~5% of these *per single look*, but peeking multiplies the chances.

**Type II error (false negative).** Burying a real winner because the test was too small. Power = 1 − Type II rate.

**Power & sample size.** Small effects + noisy traffic = you need thousands of users. Plan n first.

**P-hacking / peeking.** Re-checking after every batch and stopping the moment p < 0.05. Each look is another lottery ticket.

**Confidence interval.** The shaded bands: range of plausible true rates. If the CI of the lift includes 0, 'no effect' is still plausible.
""")


# ---------- main ----------
st.title("🎰 A/B Test Tycoon: App & Casino Designer")
with st.expander("📖 How to Play & Learning Objectives"):
    st.markdown("""**You're the Lead PM at a mobile gaming studio; your CEO is impatient.**
1. Add traffic (➕ buttons). Watch the chart, evidence meter, power meter and the 🧠 Coach.
2. 🚀 **Ship B** or 🗑️ **Reject B**: money and reputation depend on the truth *and* your evidence.
3. Survive 3 levels, earn badges, then explore the 🧪 Lab tab.

**Learning objectives:** hypothesis testing · p-value (< 0.05) · power & sample size · p-hacking (peeking) · Type I vs Type II errors.""")
t1, t2, t3 = st.tabs(["🎮 Play", "🧪 Lab", "📚 Learn"])
with t1:
    play_tab()
with t2:
    lab_tab()
with t3:
    learn_tab()
