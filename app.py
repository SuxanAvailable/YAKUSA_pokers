"""A/B Test Tycoon: App & Casino Designer (IDSE mini-project)
Learn hypothesis testing, p-values, power and p-hacking by playing.
All numbers come from live simulation (np.random.binomial) + a two-proportion z-test.
"""
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@400;600&display=swap');
.stApp{background:radial-gradient(ellipse at 20% 0%,#2a1560 0%,#0b0b16 55%),#0b0b16;font-family:Inter,sans-serif}
h2,h3{color:#ffd84d!important}
.hero{text-align:center;margin:-8px 0 4px}
.hero h1{font-family:Orbitron,sans-serif;font-weight:800;font-size:2.3rem;margin:0;background:linear-gradient(90deg,#4dd8ff,#c89bff,#ff4d8d,#ffd84d);-webkit-background-clip:text;-webkit-text-fill-color:transparent;filter:drop-shadow(0 0 14px #8a5cff88)}
.hero p{color:#a99cff;letter-spacing:.3em;font-size:.72rem;margin:0}
.hud{display:flex;gap:10px;margin:4px 0 12px}
.hud div{flex:1;background:linear-gradient(160deg,#241a52cc,#120d2acc);border:1px solid #6c3cff;border-radius:14px;padding:8px 12px;text-align:center;box-shadow:0 0 14px #6c3cff55}
.hud b{display:block;font-family:Orbitron,sans-serif;font-size:1.35rem;color:#ffd84d}.hud span{font-size:.68rem;color:#a99cff;letter-spacing:.1em}
.stButton>button{border-radius:12px;border:1px solid #6c3cff;background:linear-gradient(180deg,#2b1f63,#1a1340);color:#fff;transition:.15s}
.stButton>button:hover{border-color:#ffd84d;box-shadow:0 0 14px #ffd84d88;transform:translateY(-1px);color:#ffd84d}
.ceo{background:#2a1530;border:2px solid #ff4d8d;border-radius:16px;padding:10px 16px;font-size:1.02rem;color:#fff}
.coach{background:#10243a;border-left:5px solid #4dd8ff;border-radius:10px;padding:10px 14px;color:#dff6ff}
.mis{background:#15122f;border:1px solid #3a2f7a;border-radius:14px;padding:10px 14px}
.m{padding:2px 0;color:#d6d0ff;font-size:.92rem}.m.next{color:#ffd84d;font-weight:700}.m.done{opacity:.5;text-decoration:line-through}
.lesson{background:linear-gradient(135deg,#2b1f63,#103a3a);border:1px solid #4dd8ff;border-radius:14px;padding:12px 16px;color:#fff;margin-top:8px}
.lbl{font-size:.8rem;color:#a99cff;margin:8px 0 2px}
.bar{background:#241f45;border-radius:10px;height:18px;position:relative;overflow:hidden}
.bar i{display:block;height:100%;border-radius:10px}
.mark{position:absolute;top:0;bottom:0;width:3px;background:#fff}
.dots{font-size:.85rem;line-height:1.15;letter-spacing:1px}
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
    ss.briefed = False


def reset_game():
    st.session_state.cash, st.session_state.rep, st.session_state.badges = 100_000, 50, []
    reset_level(0)


if "level" not in st.session_state:
    st.session_state.show_tut = True
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


# ---------- tutorial clip + 3D arena ----------
TUTORIAL_HTML = """
<div style="font-family:Arial,sans-serif;background:#12102a;border:2px solid #6c3cff;border-radius:16px;padding:12px;color:#fff;box-shadow:0 0 24px #6c3cff88">
<canvas id="cv" width="640" height="220" style="width:100%;background:#0b0b16;border-radius:10px"></canvas>
<div id="cap" style="font-size:17px;margin:10px 0 6px;min-height:56px"></div>
<div style="background:#241f45;height:6px;border-radius:4px"><div id="pg" style="height:6px;width:0;background:#ffd84d;border-radius:4px"></div></div>
<div style="display:flex;justify-content:space-between;font-size:12px;color:#a99cff;margin-top:6px"><span id="stp"></span><span style="cursor:pointer" onclick="si=0;t0=performance.now()">replay</span></div></div>
<script>
const cv=document.getElementById("cv"),g=cv.getContext("2d"),cap=document.getElementById("cap"),pg=document.getElementById("pg"),stp=document.getElementById("stp");
const T=[
"<b>1. You are the Product Manager.</b> Your CEO wants every feature shipped. Your job: decide <b>Ship</b> or <b>Reject</b> using DATA, not pressure.",
"<b>2. Press the + buttons to bring users.</b> Half see version A, half see B. Each green square is a user who converted.",
"<b>3. Watch the shaded bands.</b> More users = narrower bands = less noise. Early on, luck can fake a winner.",
"<b>4. The evidence meter</b> fills as p-value falls. Crossing the white line means p &lt; 0.05: the gap is unlikely to be pure luck.",
"<b>5. Careful: do not peek and stop!</b> If p dips under 0.05 early, it may be a fluke. Then press Ship or Reject. Money and reputation follow."];
let si=0,t0=performance.now();const DUR=5200;
function sq(x,y,on,c){g.fillStyle=on?c:"#2a2550";g.fillRect(x,y,14,14);}
function draw(k,i){g.clearRect(0,0,640,220);g.font="bold 15px Arial";
 if(i==0){g.fillStyle="#ffd84d";g.font="bold 60px Arial";g.fillText("🧑‍💼",80,110);g.fillText(k>0.4?"😤":"😎",460,110);g.font="bold 18px Arial";g.fillStyle="#fff";g.fillText(k>0.4?"SHIP IT NOW!!!":"Ship the neon button",380,150);g.fillStyle="#4dd8ff";g.fillText("You: Let me check the data first...",60,170);}
 if(i==1){for(let v=0;v<2;v++){const r=v?0.5:0.3;for(let j=0;j<40;j++){const x=60+v*300+(j%10)*18,y=40+Math.floor(j/10)*18;sq(x,y,j<40*k&&(j*0.618%1)<r*1.6,v?"#ff4d8d":"#4dd8ff");}g.fillStyle=v?"#ff4d8d":"#4dd8ff";g.fillText(v?"B (new)":"A (old)",60+v*300,30);}g.fillStyle="#ffd84d";g.fillText("+200 users",270,200);}
 if(i==2){const w=70*(1-k)+8;for(let v=0;v<2;v++){const c=v?"#ff4d8d":"#4dd8ff",m=v?110:140;g.fillStyle=c+"44";g.beginPath();g.moveTo(40,m-w);g.lineTo(600,m-8);g.lineTo(600,m+8);g.lineTo(40,m+w);g.fill();g.strokeStyle=c;g.beginPath();g.moveTo(40,m+Math.sin(k*9+v)*20*(1-k));g.lineTo(40+560*k,m);g.stroke();}g.fillStyle="#fff";g.fillText("users  ->",520,205);}
 if(i==3){g.fillStyle="#241f45";g.fillRect(60,90,520,28);const f=Math.min(1,k*1.1);g.fillStyle=f>0.5?"#3ddc84":"#ff5c5c";g.fillRect(60,90,520*f*0.85,28);g.fillStyle="#fff";g.fillRect(60+520*0.34,80,4,48);g.fillText("p = 0.05",60+520*0.34-30,72);if(f>0.5){g.fillStyle="#3ddc84";g.font="bold 22px Arial";g.fillText("SIGNIFICANT!",230,170);}}
 if(i==4){g.strokeStyle="#c89bff";g.lineWidth=2;g.beginPath();for(let x=0;x<=1;x+=0.02){const y=110+Math.sin(x*14)*40*(1-x*0.6);if(x==0)g.moveTo(40,y);else g.lineTo(40+x*400*k*2.5>520?560:40+x*520,y);}g.stroke();g.strokeStyle="#ff5c5c";g.setLineDash([6,4]);g.beginPath();g.moveTo(40,150);g.lineTo(560,150);g.stroke();g.setLineDash([]);g.fillStyle="#ff5c5c";g.fillText("0.05 line",480,168);g.fillStyle="#fff";g.fillText("a dip below it can be a fluke",60,40);g.fillStyle="#2b1f63";g.fillRect(380,180,100,30);g.fillRect(500,180,100,30);g.fillStyle="#fff";g.fillText("Ship",410,200);g.fillText("Reject",522,200);}
}
function loop(t){const e=t-t0;if(e>DUR){si=(si+1)%T.length;t0=t;}const k=Math.min(1,(t-t0)/DUR);cap.innerHTML=T[si];pg.style.width=((si+k)/T.length*100)+"%";stp.textContent="Step "+(si+1)+" of "+T.length;draw(k,si);requestAnimationFrame(loop);}
requestAnimationFrame(loop);
</script>
"""

ARENA_HTML = """
<div id="c" style="width:100%;height:380px;border-radius:14px;overflow:hidden;position:relative;background:radial-gradient(circle,#1b1240,#0b0b16)">
<div style="position:absolute;left:10px;top:8px;color:#a99cff;font:12px Arial;z-index:2">drag to rotate | pillar = conversion rate | glass shell = 95% CI | orb = evidence</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D=__DATA__;const box=document.getElementById("c");
if(typeof THREE==="undefined"){box.innerHTML="<p style='color:#fff;padding:20px;font-family:Arial'>3D view needs internet access (three.js). Use the Charts tab.</p>";}
else{
const W=box.clientWidth||640,H=380;
const R=new THREE.WebGLRenderer({antialias:true,alpha:true});R.setSize(W,H);R.setPixelRatio(Math.min(2,window.devicePixelRatio||1));box.appendChild(R.domElement);
const sc=new THREE.Scene();sc.fog=new THREE.Fog(0x0b0b16,9,24);
const cam=new THREE.PerspectiveCamera(45,W/H,0.1,100);
sc.add(new THREE.AmbientLight(0x9988ff,0.7));
const L1=new THREE.PointLight(0xffffff,1.1);L1.position.set(4,7,5);sc.add(L1);
const L2=new THREE.PointLight(0xff4d8d,0.8);L2.position.set(-5,3,-3);sc.add(L2);
const S=14;
const floor=new THREE.Mesh(new THREE.CylinderGeometry(4.4,4.4,0.15,64),new THREE.MeshStandardMaterial({color:0x1a1340,emissive:0x2a1a6a,metalness:0.6,roughness:0.4}));sc.add(floor);
const rim=new THREE.Mesh(new THREE.TorusGeometry(4.4,0.05,8,96),new THREE.MeshBasicMaterial({color:0x8a5cff}));rim.rotation.x=Math.PI/2;rim.position.y=0.08;sc.add(rim);
function label(txt,col,w){const c=document.createElement("canvas");c.width=360;c.height=80;const x=c.getContext("2d");x.font="bold 46px Arial";x.textAlign="center";x.fillStyle=col;x.fillText(txt,180,56);const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));s.scale.set(w||2.6,0.58,1);return s;}
function pillar(x,col){
 const m=new THREE.Mesh(new THREE.CylinderGeometry(0.75,0.75,1,40),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.35,metalness:0.4,roughness:0.3}));m.position.x=x;sc.add(m);
 const c=new THREE.Mesh(new THREE.CylinderGeometry(1.0,1.0,1,40,1,true),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.22,side:THREE.DoubleSide}));c.position.x=x;sc.add(c);
 return [m,c];}
const PA=pillar(-1.8,0x4dd8ff),PB=pillar(1.8,0xff4d8d);
const la=label("A  "+(D.ra*100).toFixed(1)+"%","#4dd8ff");la.position.set(-1.8,0.45,1.9);sc.add(la);
const lb=label("B  "+(D.rb*100).toFixed(1)+"%","#ff4d8d");lb.position.set(1.8,0.45,1.9);sc.add(lb);
const col=D.sig?0x3ddc84:0xff5c5c;
const orb=new THREE.Mesh(new THREE.SphereGeometry(0.6,32,32),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:0.9}));orb.position.set(0,4.4,0);sc.add(orb);
const ring=new THREE.Mesh(new THREE.TorusGeometry(1.0,0.04,12,64),new THREE.MeshBasicMaterial({color:0xffd84d}));ring.position.set(0,4.4,0);sc.add(ring);
const lp=label(D.n?("p = "+D.p.toFixed(4)):"p = ?",D.sig?"#3ddc84":"#ff9a9a",2.8);lp.position.set(0,5.5,0);sc.add(lp);
if(D.truth){[[-1.8,D.truth[0]],[1.8,D.truth[1]]].forEach(function(a){const t=new THREE.Mesh(new THREE.TorusGeometry(1.05,0.05,8,48),new THREE.MeshBasicMaterial({color:0xffd84d}));t.rotation.x=Math.PI/2;t.position.set(a[0],a[1]*S+0.08,0);sc.add(t);});}
const parts=[];
if(D.sig){for(let i=0;i<36;i++){const p=new THREE.Mesh(new THREE.SphereGeometry(0.07,8,8),new THREE.MeshBasicMaterial({color:0xffd84d}));p.position.set((Math.random()-0.5)*6,Math.random()*6,(Math.random()-0.5)*4);sc.add(p);parts.push(p);}}
function upd(a,r,c,k){const h=Math.max(0.03,r*S*k);a[0].scale.y=h;a[0].position.y=h/2+0.08;const lo=Math.max(0,r-c)*S*k,hi=(r+c)*S*k,ch=Math.max(0.03,hi-lo);a[1].scale.y=ch;a[1].position.y=lo+ch/2+0.08;}
let k=0,az=0.6,el=0.35,drag=false,lx=0,ly=0;
R.domElement.addEventListener("pointerdown",function(e){drag=true;lx=e.clientX;ly=e.clientY;});
window.addEventListener("pointerup",function(){drag=false;});
window.addEventListener("pointermove",function(e){if(drag){az-=(e.clientX-lx)*0.01;el=Math.max(0.1,Math.min(1.2,el+(e.clientY-ly)*0.005));lx=e.clientX;ly=e.clientY;}});
function frame(t){requestAnimationFrame(frame);k+=(1-k)*0.05;if(!drag)az+=0.003;
 upd(PA,D.ra,D.ca,k);upd(PB,D.rb,D.cb,k);
 const s=(0.5+0.8*D.ev)*(1+0.06*Math.sin(t*0.004));orb.scale.set(s,s,s);ring.rotation.x=t*0.001;ring.rotation.y=t*0.0013;ring.scale.set(s,s,s);
 parts.forEach(function(p){p.position.y+=0.03;if(p.position.y>6.5)p.position.y=0.2;});
 const Rr=10.5;cam.position.set(Rr*Math.sin(az)*Math.cos(el),Rr*Math.sin(el)+1.5,Rr*Math.cos(az)*Math.cos(el));cam.lookAt(0,2.2,0);R.render(sc,cam);}
requestAnimationFrame(frame);
}
</script>
"""


def embed(html, height):
    """Render an HTML/JS snippet (uses st.iframe on new Streamlit, components.html on old)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def arena(L, reveal):
    n = ss.na
    ra, rb = (ss.xa / n, ss.xb / n) if n else (0.0, 0.0)
    ca = 1.96 * np.sqrt(ra * (1 - ra) / n) if n else 0.0
    cb = 1.96 * np.sqrt(rb * (1 - rb) / n) if n else 0.0
    _, p, _ = cur()
    d = dict(ra=float(ra), rb=float(rb), ca=float(ca), cb=float(cb), p=float(p), n=int(n),
             ev=float(min(1, -np.log10(max(p, 1e-4)) / 4)) if n else 0.0,
             sig=bool(n and p < ALPHA and ss.xb > ss.xa), truth=list(reveal) if reveal else None)
    embed(ARENA_HTML.replace("__DATA__", json.dumps(d)), 390)


BRIEF = [
    dict(goal="Decide whether to ship the Neon Spin Button.",
         win="Ship B only after the 🎯 evidence meter crosses the white line (p < 0.05).",
         watch="The glass shells (95% CI) shrinking, and the evidence orb turning green.",
         concept="Signal vs noise: with enough users a real gap rises above random noise."),
    dict(goal="Decide on the Double XP Pop-up. The CEO will scream SHIP every time p dips under 0.05.",
         win="Collect 1,000+ users per variant before you decide. Don't be fooled by an early dip.",
         watch="The 👀 peek counter and how the p-value line wanders up and down.",
         concept="P-hacking: every peek is another lottery ticket for a false positive."),
    dict(goal="Decide on the Golden Loot Box. If it helps, the lift is small and the traffic is noisy.",
         win="Collect thousands of users per variant until the 🔋 power meter nears 80%.",
         watch="The power meter: low power means real effects stay hidden.",
         concept="Statistical power: small effects need big samples, or you get Type II errors."),
]
LESSONS = [
    "**Concept unlocked: p-value & signal vs noise.** p < 0.05 means: if the button did nothing, a gap this big would appear by luck under 5% of the time. About 700 users per variant exposed the real +5-pt lift; 50 users could not.",
    "**Concept unlocked: p-hacking.** A and B were identical. Each peek gives noise a fresh chance to dip under 0.05, so stopping at the first dip gives far more than 5% false alarms (see the 🧪 Lab). Fix: choose the sample size first, then look once.",
    "**Concept unlocked: power.** The lift was real but small (+1.5 pts). Power stays low until roughly 6,500 users per variant, so small tests often say 'not significant' (Type II error). Try the Lab's sample-size calculator.",
]


@st.dialog("🎬 How to play (about 30 seconds)", width="large")
def tutorial_dialog():
    embed(TUTORIAL_HTML, 380)
    c1, c2 = st.columns(2)
    if c1.button("⏭ Skip tutorial", use_container_width=True):
        st.rerun()
    if c2.button("▶ Start playing", type="primary", use_container_width=True):
        st.rerun()


@st.dialog("📋 Mission briefing", width="large")
def briefing():
    L, B = LEVELS[ss.level], BRIEF[ss.level]
    st.markdown(f"### {L['icon']} {L['title']}\n{L['brief']}")
    st.markdown(f"**🎯 Goal:** {B['goal']}\n\n**✅ How to win:** {B['win']}\n\n**👀 Watch:** {B['watch']}\n\n**💡 Concept you'll learn:** {B['concept']}")
    if st.button("▶ Start mission", type="primary"):
        st.rerun()


def replay_tut():
    ss.show_tut = True


def missions():
    h, n = ss.hist, ss.na
    anysig = any(x[3] < ALPHA for x in h)
    done = ss.result is not None
    M = [
        [("Add 200+ users per variant (➕200)", n >= 200), ("Reach 500+ and watch the glass shells shrink", n >= 500),
         ("Wait until the evidence orb turns green (p < 0.05)", anysig), ("Make your call: Ship or Reject", done)],
        [("Add users in small batches (3+ clicks of ➕50 / ➕200)", len(h) >= 3), ("Spot a moment where p dips below 0.05 👀", anysig),
         ("Resist! Collect 1,000+ users per variant", n >= 1000), ("Make your call", done)],
        [("Add 200+ users and see it look unconvincing", n >= 200), ("Collect 3,000+ users per variant", n >= 3000),
         ("Push to 6,500+ so power reaches ~80%", n >= 6500), ("Make your call", done)],
    ][ss.level]
    nxt = next((i for i, (_, ok) in enumerate(M) if not ok), None)
    rows = "".join(f'<div class="m {"done" if ok else ("next" if i == nxt else "")}">{"✅" if ok else ("👉" if i == nxt else "⬜")} {t}</div>'
                   for i, (t, ok) in enumerate(M))
    st.markdown(f'<div class="mis"><b>🎯 MISSION CHECKLIST</b>{rows}</div>', unsafe_allow_html=True)


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
    m1, m2 = st.columns(2)
    with m1:
        missions()
    with m2:
        if ss.result is None:
            face, line = ceo()
            st.markdown(f'<div class="ceo">{face} <b>CEO:</b> {line}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="coach" style="margin-top:8px">🧠 <b>Coach:</b> {coach()}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="lesson">💡 {LESSONS[ss.level]}</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        rv = (L["base"], L["base"] + L["lift"]) if ss.result else None
        v3, v2 = st.tabs(["🧊 3D Arena", "📈 Charts"])
        with v3:
            arena(L, rv)
        with v2:
            draw_chart(rv)
            if ss.last:
                n, ca, cb = ss.last
                st.caption(f"Latest batch: +{n:,} users per variant (60 sampled shown) 🟩 converted ⬛ didn't")
                c1, c2 = st.columns(2)
                c1.markdown(f'<div class="dots">A<br>{dots(ca / n)}</div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="dots">B<br>{dots(cb / n)}</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f"**Step 1. Bring users** (now: `{ss.na:,}` per variant)")
        for col, n in zip(st.columns(4), BATCHES):
            col.button(f"➕{n:,}", key=f"b{n}", on_click=run_batch, args=(n,), disabled=ss.result is not None, use_container_width=True)
        _, p, (lo, hi) = cur()
        ra, rb = (ss.xa / ss.na, ss.xb / ss.na) if ss.na else (0.0, 0.0)
        st.markdown(f'<div class="hud"><div><b>{ra * 100:.2f}%</b><span>A RATE</span></div>'
                    f'<div><b>{rb * 100:.2f}%</b><span>B RATE ({(rb - ra) * 100:+.2f})</span></div></div>', unsafe_allow_html=True)
        st.markdown("**Step 2. Read the evidence**")
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
            st.markdown("**Step 3. Make the call**")
            d1, d2 = st.columns(2)
            d1.button("🚀 Ship B", on_click=decide, args=(True,), disabled=ss.na == 0, use_container_width=True)
            d2.button("🗑️ Reject B", on_click=decide, args=(False,), disabled=ss.na == 0, use_container_width=True)
            st.button("↩️ Restart level", on_click=reset_level, args=(ss.level,))
        else:
            r = ss.result
            {"good": st.success, "bad": st.error, "meh": st.info}[r["kind"]](
                f"**{r['title']}**\n\n{r['text']}\n\nCash {r['dcash']:+,} | Reputation {r['drep']:+d}")
            st.caption(f"🔍 Truth revealed: A = {L['base'] * 100:.1f}%, B = {(L['base'] + L['lift']) * 100:.1f}% (gold rings / dashed lines)")
            st.button("Next ➡️" if ss.level < 2 else "🏁 Final score", on_click=go_next, type="primary")


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
st.markdown('<div class="hero"><h1>A/B TEST TYCOON</h1><p>APP &amp; CASINO DESIGNER</p></div>', unsafe_allow_html=True)
if ss.show_tut:
    ss.show_tut = False
    tutorial_dialog()
elif not ss.briefed and ss.level < 3:
    ss.briefed = True
    briefing()
c_a, c_b = st.columns([4, 1])
with c_a:
    with st.expander("📖 How to Play & Learning Objectives"):
        st.markdown("""**You're the Lead PM at a mobile gaming studio; your CEO is impatient.**
1. Follow the 🎯 **mission checklist**: it always shows your next step.
2. Bring users (➕), read the 🎯 evidence meter and 🔋 power meter, listen to the 🧠 Coach.
3. 🚀 **Ship B** or 🗑️ **Reject B**: money and reputation depend on the truth *and* your evidence. Each level ends with a 💡 concept card.
4. Finish 3 levels, collect badges, then explore the 🧪 Lab tab.

**Learning objectives:** hypothesis testing · p-value (< 0.05) · power & sample size · p-hacking (peeking) · Type I vs Type II errors.""")
c_b.button("🎬 Replay tutorial", on_click=replay_tut, use_container_width=True)
t1, t2, t3 = st.tabs(["🎮 Play", "🧪 Lab", "📚 Learn"])
with t1:
    play_tab()
with t2:
    lab_tab()
with t3:
    learn_tab()
